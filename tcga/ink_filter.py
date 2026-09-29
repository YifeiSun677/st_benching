#!/usr/bin/env python
"""TCGA-BRCA arm, step 2b -- drop spots that sit on ink (run once, after prep_slides.py).

prep_slides.py removes black ink from the 4-um tissue mask as  V < 60 AND S < 100.  Near-black
pixels break that rule: at V ~ 10 HSV saturation is numerically unstable and reads 150-180, so
surgical-margin ink on tissue stayed in the mask (TCGA-D8-A13Y DX1 w10: a whole inked strip got
spots).  This step re-checks every exported spot on the full-resolution her2st-scale JPEG:

  ink pixel   (V < 60 AND S < 100)  OR  V < 30          (densely stained frozen tissue: V < 60 but
                                                          S 124-213 and V well above 30 -> not ink)
  ink_frac    fraction of ink pixels in the spot's 100-um disk
  drop        ink_frac >= 0.5   (mirrors MIN_TISSUE = 0.5)

Spot files and placeholder counts are rewritten without the dropped spots (the image is not touched);
a window left empty is removed.  Adds column ink_frac to the spot files.  Idempotent: a spot file
that already has ink_frac is skipped.

writes meta/ink_filter.tsv (section, n_before, n_dropped) and updates meta/slides/*.json counts
usage: python tcga/ink_filter.py [--dry-run] [--sections SEC ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCI = Path.home() / "Documents" / "science"
V_DARK, S_MAX, V_BLACK, DROP = 60, 100, 30, 0.5


def ink_fracs(img_bgr: np.ndarray, px, py, r: float) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    ink = ((V < V_DARK) & (S < S_MAX)) | (V < V_BLACK)
    k = int(np.ceil(r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = (xx ** 2 + yy ** 2) <= r * r
    out = np.empty(len(px))
    for i, (x, y) in enumerate(zip(np.round(px).astype(int), np.round(py).astype(int))):
        win = ink[y - k:y + k + 1, x - k:x + k + 1]
        out[i] = win[disk[:win.shape[0], :win.shape[1]]].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCI / "tcga_upload"))
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    data = out / "data"
    params = json.loads((out / "meta" / "params.json").read_text())
    r = params["SPOT_UM"] / 2 / params["her2st_scale"]["um_per_px_ref"]        # spot radius, her2st px
    files = sorted((data / "ST-spotfiles").glob("*_selection.tsv"))
    if a.sections:
        files = [f for f in files if f.name[:-len("_selection.tsv")] in a.sections]
    rows = []
    for f in files:
        sec = f.name[:-len("_selection.tsv")]
        sp = pd.read_csv(f, sep="\t")
        if "ink_frac" in sp and not a.dry_run:
            continue
        patient = sec.split("_")[0]
        img = cv2.imread(str(data / "ST-imgs" / patient / sec / f"HE_{sec}.jpg"))
        sp["ink_frac"] = ink_fracs(img, sp.pixel_x.values, sp.pixel_y.values, r)
        drop = sp.ink_frac >= DROP
        rows.append(dict(section=sec, n_before=len(sp), n_dropped=int(drop.sum())))
        if drop.any():
            print(f"{sec}: {int(drop.sum())}/{len(sp)} spots on ink", flush=True)
        if a.dry_run:
            continue
        keep = sp[~drop]
        cnt = data / "ST-cnts" / f"{sec}.tsv.gz"
        if keep.empty:
            f.unlink()
            cnt.unlink()
            shutil.rmtree(data / "ST-imgs" / patient / sec)
            continue
        keep.to_csv(f, sep="\t", index=False, float_format="%.4f")
        c = pd.read_csv(cnt, sep="\t", index_col=0)
        ids = [f"{int(x)}x{int(y)}" for x, y in zip(keep.x, keep.y)]
        c.loc[ids].to_csv(cnt, sep="\t", compression="gzip")
    df = pd.DataFrame(rows)
    print(f"{len(df)} sections checked, {int(df.n_dropped.sum()) if len(df) else 0} spots on ink, "
          f"{int((df.n_dropped == df.n_before).sum()) if len(df) else 0} sections emptied")
    if a.dry_run or df.empty:
        return
    log = out / "meta" / "ink_filter.tsv"
    if log.exists():
        df = pd.concat([pd.read_csv(log, sep="\t"), df]).drop_duplicates("section", keep="last")
    df.to_csv(log, sep="\t", index=False)
    # refresh per-slide records so windows.tsv / packing see the new spot counts
    left = {p.name[:-len("_selection.tsv")]: len(pd.read_csv(p, sep="\t"))
            for p in (data / "ST-spotfiles").glob("*_selection.tsv")}
    for j in (out / "meta" / "slides").glob("*.json"):
        rec = json.loads(j.read_text())
        if rec.get("status") != "ok":
            continue
        ws = []
        for w in rec["windows"]:
            if w["section"] in left:
                w["n_spots"] = left[w["section"]]
                ws.append(w)
        rec["windows"], rec["n_windows"] = ws, len(ws)
        rec["n_spots"] = sum(w["n_spots"] for w in ws)
        j.write_text(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
