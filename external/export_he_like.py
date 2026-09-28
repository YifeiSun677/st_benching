#!/usr/bin/env python
"""He Stage 2 -- resample each He section to her2st's image scale and write it in the her2st
file layout under /workspace/ext/he/her2st_like/data (NOT the her2st tree).

  scale   He um/px from its own spot grid (pixel ~ array unit, 200 um pitch, per section);
          her2st reference um/px from ext/calib/her2st_scale.json (calib_her2st_scale.py, the
          same file the Visium arm used).  f = um/px_he / um/px_her2st;  image and spot pixels * f.
  spots   under-tissue spots from the spots file, with a count row, > 0 UMI.

  ST-imgs/<patient>/<SEC>/HE_<SEC>.jpg    resampled H&E (JPEG Q95, as the Visium arm)
  ST-spotfiles/<SEC>_selection.tsv        x, y (= new_x, new_y; He is already in 200-um array units),
                                          pixel_x, pixel_y (resampled image; x = column), selected,
                                          tumor (1/0/-1 = tumour / non / unlabelled)
  ST-cnts/<SEC>.tsv.gz                    raw counts, spots x SYMBOL, genes with > 0 counts only
  calib/counts_<SEC>.npz                  sparse raw counts, all ENSG features, same row order
                                          (genes = symbol, gene_ids = ENSG)
  calib/panel_coverage.tsv                section, gene, measured (symbol in ANY He section), detected
  calib/he_scale.tsv                      scale bookkeeping per section
  calib/qc/<SEC>.jpg                      spot centres + 224-px window drawn on the resampled image

usage: python external/export_he_like.py [SEC ...]      (default: all sections in metadata)
needs: opencv-python, pillow, scipy;  ST-Net checkout at $STNET (default /workspace/ST-Net) for
       ST-Net's own ENSG -> symbol table
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common as K
import he_common as H

Image.MAX_IMAGE_PIXELS = None
WIN = 224


def export(sec, ref, symbol, a):
    t0 = time.time()
    p = H.raw_paths(sec)
    sp = H.read_raw_spots(p["spots"])
    cnt = H.read_raw_counts(p["counts"])
    tum = H.read_tumor(p["tumor"])

    sc = H.fit_scale(sp)
    if (sc["xy_diff_pct"] > 3 or sc["rmse_frac_pitch"] > 0.10) and not a.force:
        raise SystemExit(f"{sec}: spot grid fit failed {sc} -- run verify_he_raw.py")
    f = sc["um_per_px"] / ref["um_per_px_ref"]

    n0 = len(sp)
    sp = sp.loc[sp.index.intersection(cnt.index)]
    cnt = cnt.loc[sp.index]
    umi = cnt.values.sum(1)
    sp, cnt = sp[umi > 0], cnt[umi > 0]
    n_drop = n0 - len(sp)

    # ---- image ------------------------------------------------------------------
    import cv2
    img = np.asarray(Image.open(p["image"]).convert("RGB"))
    Hh, W = img.shape[:2]
    newW, newH = int(round(W * f)), int(round(Hh * f))
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LANCZOS4
    small = cv2.resize(img, (newW, newH), interpolation=interp)
    del img
    idir = H.HE_DATA / "ST-imgs" / p["patient"] / sec
    idir.mkdir(parents=True, exist_ok=True)
    for old in idir.glob("*"):
        old.unlink()                                   # ports read the FIRST file in this dir
    cv2.imwrite(str(idir / f"HE_{sec}.jpg"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    px, py = sp.pixel_x.values * f, sp.pixel_y.values * f
    assert px.max() < newW and py.max() < newH and px.min() >= 0 and py.min() >= 0
    edge = int(((px < WIN / 2) | (py < WIN / 2) | (px > newW - WIN / 2) | (py > newH - WIN / 2)).sum())

    # ---- spot file ----------------------------------------------------------------
    lab = np.array([{"tumor": 1, "non": 0}.get(tum.get(s, ""), -1) for s in sp.index])
    spot = pd.DataFrame({"x": sp.x.values, "y": sp.y.values, "new_x": sp.x.values, "new_y": sp.y.values,
                         "pixel_x": px, "pixel_y": py, "selected": 1, "tumor": lab,
                         "pixel_x_raw": sp.pixel_x.values, "pixel_y_raw": sp.pixel_y.values})
    (H.HE_DATA / "ST-spotfiles").mkdir(parents=True, exist_ok=True)
    spot.to_csv(H.HE_DATA / "ST-spotfiles" / f"{sec}_selection.tsv", sep="\t", index=False, float_format="%.4f")

    # ---- counts -----------------------------------------------------------------------
    ensg = list(cnt.columns)
    sym = [str(symbol[g]) for g in ensg]
    Xs = sparse.csr_matrix(cnt.values)
    np.savez_compressed(H.HE_CALIB / f"counts_{sec}.npz", data=Xs.data, indices=Xs.indices,
                        indptr=Xs.indptr, shape=Xs.shape, genes=np.array(sym), gene_ids=np.array(ensg),
                        spot_id=np.array(list(sp.index)))
    dense = pd.DataFrame(cnt.values, index=sp.index, columns=sym)
    dense = dense.loc[:, dense.sum(0) > 0]
    n_dup = int(dense.columns.duplicated().sum())
    if n_dup:
        dense = dense.T.groupby(level=0).sum().T          # duplicate symbols -> summed
    (H.HE_DATA / "ST-cnts").mkdir(parents=True, exist_ok=True)
    dense.to_csv(H.HE_DATA / "ST-cnts" / f"{sec}.tsv.gz", sep="\t", compression="gzip")

    # ---- QC overlay -------------------------------------------------------------------------
    qc = small.copy()
    for x, y, t in zip(px.astype(int), py.astype(int), lab):
        col = (220, 30, 30) if t == 1 else (30, 30, 220) if t == 0 else (30, 160, 30)
        cv2.circle(qc, (x, y), max(3, int(0.25 * ref["px_per_unit_ref"])), col, 3)
    x0, y0 = int(np.median(px)), int(np.median(py))
    cv2.rectangle(qc, (x0 - WIN // 2, y0 - WIN // 2), (x0 + WIN // 2, y0 + WIN // 2), (0, 0, 0), 4)
    s = 1600 / max(qc.shape[:2])
    qc = cv2.resize(qc, (int(qc.shape[1] * s), int(qc.shape[0] * s)), interpolation=cv2.INTER_AREA)
    (H.HE_CALIB / "qc").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(H.HE_CALIB / "qc" / f"{sec}.jpg"), cv2.cvtColor(qc, cv2.COLOR_RGB2BGR))

    row = dict(section=sec, patient=p["patient"], subtype=p["subtype"], um_per_px_he=sc["um_per_px"],
               px_per_unit_he=sc["px_per_unit"], xy_diff_pct=sc["xy_diff_pct"], rmse_frac_pitch=sc["rmse_frac_pitch"],
               um_per_px_her2st=ref["um_per_px_ref"], f=f, img_in=f"{W}x{Hh}", img_out=f"{newW}x{newH}",
               n_spots=len(spot), n_dropped=n_drop, n_edge_spots=edge, n_tumor=int((lab == 1).sum()),
               n_dup_symbols=n_dup, genes_written=dense.shape[1], seconds=round(time.time() - t0))
    print(json.dumps(row, default=float), flush=True)
    return row, set(sym), set(dense.columns)


def upsert(path, df, key="section"):
    if path.exists():
        old = pd.read_csv(path, sep="\t")
        df = pd.concat([old[~old[key].isin(df[key].unique())], df], ignore_index=True)
    df.to_csv(path, sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*")
    ap.add_argument("--force", action="store_true", help="ignore the grid-fit sanity check")
    a = ap.parse_args()
    secs = a.sections or H.he_sections()
    H.HE_CALIB.mkdir(parents=True, exist_ok=True)
    refp = K.CALIB / "her2st_scale.json"
    if not refp.exists():
        raise SystemExit(f"{refp} missing -- run: python external/calib_her2st_scale.py")
    ref = json.loads(refp.read_text())
    symbol = H.ensembl_symbols()
    panel = K.load_panel()
    print(f"her2st ref {ref['um_per_px_ref']:.4f} um/px ({ref['px_per_unit_ref']:.2f} px per 200 um)")

    rows, detected = [], {}
    for sec in secs:
        row, feats, det = export(sec, ref, symbol, a)
        rows.append(row)
        detected[sec] = det
    upsert(H.HE_CALIB / "he_scale.tsv", pd.DataFrame(rows))

    # measured = symbol present in ANY He section (rebuilt from all exported npz, not just this call)
    allfeat = set()
    for f in sorted(H.HE_CALIB.glob("counts_*.npz")):
        allfeat |= {str(g) for g in np.load(f, allow_pickle=True)["genes"]}
    cov = []
    for f in sorted((H.HE_DATA / "ST-cnts").glob("*.tsv.gz")):
        sec = f.name[:-len(".tsv.gz")]
        det = detected.get(sec) or set(pd.read_csv(f, sep="\t", index_col=0, nrows=0).columns)
        cov.append(pd.DataFrame({"section": sec, "gene": panel, "measured": [int(g in allfeat) for g in panel],
                                 "detected": [int(g in det) for g in panel]}))
    cov = pd.concat(cov)
    cov.to_csv(H.HE_CALIB / "panel_coverage.tsv", sep="\t", index=False)
    m = cov.drop_duplicates("gene").measured.sum()
    print(f"panel genes present in the He feature universe: {m}/{len(panel)}; "
          f"not present: {sorted(set(panel) - allfeat)[:20]}")


if __name__ == "__main__":
    main()
