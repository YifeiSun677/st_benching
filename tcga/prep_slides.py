#!/usr/bin/env python
"""TCGA-BRCA arm, step 2 -- slides -> her2st-like virtual sections (CPU only; runs on the Mac).

Per slide (streamed: download -> process -> delete the .svs, so disk use stays small):
  1  download from the GDC data endpoint, check md5
  2  read mpp / objective / vendor.  No mpp -> status 'no_mpp' (patient replaced from reserve)
  3  tissue mask at MASK_UM um/px: HSV saturation > max(Otsu, S_MIN), minus pen ink
     (green hue; black = dark AND unsaturated -- densely stained frozen tissue is dark but
     saturated purple and stays tissue), open/close, drop components < MIN_COMP_MM2.
     No blue-ink rule: its hue overlaps bluish haematoxylin, and on real slides it removed tumour
     nests (TCGA-BH-A18H DX1: 8.8 mm2).  Blue ink, when present, is left in the mask.
  4  virtual legacy-ST grid, PITCH_UM pitch, origin = tissue bbox corner; keep a spot when
     >= MIN_TISSUE of its SPOT_UM-diameter disk is tissue
  5  every kept spot goes into exactly one window (the whole slide is predicted):
       a. spots are grouped into tissue clusters (grid cells 8-connected after a 1-cell dilation,
          i.e. pieces separated by >= 2 empty grid steps are separate clusters)
       b. each cluster's bbox is split into windows of at most WIN_X x WIN_Y spots (her2st array
          size).  Balanced split: n = ceil(extent / WIN), size = ceil(extent / n) -> no slivers
       c. a window with < MIN_WIN_SPOTS spots joins the nearest other window when the merged bbox
          stays <= MAX_WIN in both axes (HisToGene/Hist2ST need array coords < 64); otherwise it
          stays on its own
  6  each window is written in the her2st layout, image resampled to her2st um/px
     (her2st_scale.json, the same reference the Visium and He arms used):
       ST-imgs/<patient>/<SEC>/HE_<SEC>.jpg   JPEG Q95 (as He).  Pixels farther than KEEP_PX
                                              (Chebyshev) from every kept spot are set white; no
                                              model reads more than 112 px from a spot centre
       ST-spotfiles/<SEC>_selection.tsv       x, y (window-local array coords, 1-based), new_x,
                                              new_y, pixel_x, pixel_y, selected, + bookkeeping
       ST-cnts/<SEC>.tsv.gz                   PLACEHOLDER counts (all 1, panel genes) so the
                                              ports' readers work.  Never a target.
     SEC = <patient>_<slide token>_w<NN>, e.g. TCGA-A2-A04U_TSA_w01
  7  QC thumbnail: mask outline, pen, kept spots coloured by window, window boxes

Outputs under --out:  data/ (her2st layout), qc/<SEC-prefix>.jpg, meta/slides/<file_id>.json
Resumable: a slide with meta/slides/<file_id>.json is skipped.  --collect writes meta/slides.tsv
and meta/windows.tsv from those json files.

usage:  python tcga/prep_slides.py --jobs 10                 (all main-cohort slides)
        python tcga/prep_slides.py --local a.svs b.svs       (test on files already on disk)
        python tcga/prep_slides.py --collect
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

cv2.setNumThreads(1)

SCI = Path.home() / "Documents" / "science"
REPO = Path(__file__).resolve().parents[1]
GDC_DATA = "https://api.gdc.cancer.gov/data/"

# ---- fixed parameters (written to meta/params.json) --------------------------------------
P = dict(
    PITCH_UM=200.0,        # legacy ST centre-to-centre
    SPOT_UM=100.0,         # legacy ST spot diameter
    WIN_X=33, WIN_Y=35,    # legacy ST array (her2st x 2..32, y 2..34)
    MASK_UM=4.0,           # tissue mask resolution
    S_MIN=15,              # floor on the Otsu saturation threshold (0-255)
    V_DARK=60,             # black ink / debris: V below this AND S below DARK_S_MAX
    DARK_S_MAX=100,
    PEN_GREEN=(35, 95, 40),    # hue lo, hue hi, min S   (OpenCV hue 0-180)
    MIN_COMP_MM2=0.02,     # drop tissue components smaller than this
    MIN_TISSUE=0.5,        # fraction of the spot disk that must be tissue
    MIN_WIN_SPOTS=20,      # smaller windows are merged into a neighbour when possible
    MAX_WIN=48,            # hard cap on a merged window's extent (array units); models need < 64
    KEEP_PX=224,           # image kept within this Chebyshev distance (her2st px) of a kept spot
    JPEG_Q=95,
    STRIP_OUT_PX=512,      # rows per level-0 read strip (memory bound)
)


def slide_token(file_name: str) -> str:
    m = re.match(r"^TCGA-\w\w-\w{4}-\d\d[A-Z]-\d\d-([A-Z]{2}[0-9A-Z]+)\.", file_name)
    return m.group(1)


# ------------------------------------------------------------------------ io ----
def download(file_id: str, md5: str, dest: Path, tries: int = 4) -> None:
    for k in range(tries):
        try:
            h = hashlib.md5()
            with requests.get(GDC_DATA + file_id, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
                        h.update(chunk)
            if h.hexdigest() == md5:
                return
            err = f"md5 mismatch {h.hexdigest()} != {md5}"
        except Exception as e:                                       # noqa: BLE001
            err = repr(e)
        dest.unlink(missing_ok=True)
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"download failed after {tries} tries: {err}")


def read_scaled(slide, x0_um, y0_um, w_out, h_out, out_um, mpp, max_level_um=None, strip=None):
    """RGB uint8 [h_out, w_out] of the slide region starting at (x0_um, y0_um) at out_um/px.
    Reads the coarsest pyramid level that is still at least as fine as out_um (or max_level_um),
    in horizontal strips so memory stays bounded.  Outside the slide = white."""
    limit = max_level_um or out_um
    lv = max([l for l in range(slide.level_count) if mpp * slide.level_downsamples[l] <= limit * 1.0001],
             default=0)
    ds = slide.level_downsamples[lv]
    lmpp = mpp * ds
    s = out_um / lmpp                                    # level px per output px
    W0, H0 = slide.dimensions
    out = np.full((h_out, w_out, 3), 255, np.uint8)
    step = strip or h_out
    for r0 in range(0, h_out, step):
        r1 = min(h_out, r0 + step)
        # level-lv pixel rectangle for output rows r0..r1
        lx0 = x0_um / lmpp
        ly0 = y0_um / lmpp + r0 * s
        lx1 = lx0 + w_out * s
        ly1 = y0_um / lmpp + r1 * s
        ix0, iy0 = int(math.floor(lx0)), int(math.floor(ly0))
        ix1, iy1 = int(math.ceil(lx1)), int(math.ceil(ly1))
        # clip to the level
        Lw, Lh = slide.level_dimensions[lv]
        cx0, cy0, cx1, cy1 = max(ix0, 0), max(iy0, 0), min(ix1, Lw), min(iy1, Lh)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        reg = slide.read_region((int(round(cx0 * ds)), int(round(cy0 * ds))), lv, (cx1 - cx0, cy1 - cy0))
        reg = np.asarray(reg)
        rgb = reg[..., :3].copy()
        rgb[reg[..., 3] == 0] = 255                          # transparent (outside scan) -> white
        canvas = np.full((iy1 - iy0, ix1 - ix0, 3), 255, np.uint8)
        canvas[cy0 - iy0:cy1 - iy0, cx0 - ix0:cx1 - ix0] = rgb
        # sub-pixel alignment: resize the integer box, which starts <= 1 level px before lx0
        res = cv2.resize(canvas, (w_out, r1 - r0), interpolation=cv2.INTER_AREA if s > 1 else cv2.INTER_LANCZOS4)
        out[r0:r1] = res
        del reg, rgb, canvas
    return out, lv


# ---------------------------------------------------------------------- mask ----
def tissue_mask(rgb: np.ndarray):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    black = (V < P["V_DARK"]) & (S < P["DARK_S_MAX"])
    valid = ~black
    otsu, _ = cv2.threshold(S[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(otsu), P["S_MIN"])
    g_lo, g_hi, g_s = P["PEN_GREEN"]
    pen = ((H >= g_lo) & (H < g_hi) & (S > g_s)) | ~valid
    pen = cv2.dilate(pen.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    m = ((S > t) & ~pen).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    min_px = P["MIN_COMP_MM2"] * 1e6 / P["MASK_UM"] ** 2
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    m = keep[lab]
    return m, pen & (S > t), t


def balanced_split(lo: int, hi: int, win: int) -> list[tuple[int, int]]:
    ext = hi - lo + 1
    n = math.ceil(ext / win)
    size = math.ceil(ext / n)
    return [(lo + k * size, min(hi, lo + (k + 1) * size - 1)) for k in range(n)]


def assign_windows(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Window label per spot (0..n-1, ordered top-left first).  See docstring step 5."""
    occ = np.zeros((gy.max() + 3, gx.max() + 3), np.uint8)
    occ[gy + 1, gx + 1] = 1
    _, lab = cv2.connectedComponents(cv2.dilate(occ, np.ones((3, 3), np.uint8)), connectivity=8)
    cl = lab[gy + 1, gx + 1]
    key = [None] * len(gx)
    for c in np.unique(cl):
        m = cl == c
        xs_w = balanced_split(int(gx[m].min()), int(gx[m].max()), P["WIN_X"])
        ys_w = balanced_split(int(gy[m].min()), int(gy[m].max()), P["WIN_Y"])
        wx = np.searchsorted([a for a, _ in xs_w], gx[m], side="right") - 1
        wy = np.searchsorted([a for a, _ in ys_w], gy[m], side="right") - 1
        for i, a, b in zip(np.nonzero(m)[0], wy, wx):
            key[i] = (int(c), int(a), int(b))
    by_key: dict = {}
    for i, k in enumerate(key):
        by_key.setdefault(k, set()).add(i)
    groups = dict(enumerate(by_key.values()))

    def bbox(idx):
        i = np.fromiter(idx, int)
        return gx[i].min(), gx[i].max(), gy[i].min(), gy[i].max()

    while True:
        small = sorted((len(v), k) for k, v in groups.items() if len(v) < P["MIN_WIN_SPOTS"])
        merged = False
        for _, k in small:
            x0, x1, y0, y1 = bbox(groups[k])
            best = None
            for j, v in groups.items():
                if j == k:
                    continue
                a0, a1, b0, b1 = bbox(v)
                if max(x1, a1) - min(x0, a0) + 1 > P["MAX_WIN"] or max(y1, b1) - min(y0, b0) + 1 > P["MAX_WIN"]:
                    continue
                d = max(0, a0 - x1, x0 - a1) + max(0, b0 - y1, y0 - b1)      # grid gap between bboxes
                if best is None or (d, -len(v)) < best[0]:
                    best = ((d, -len(v)), j)
            if best is not None:
                groups[best[1]] |= groups.pop(k)
                merged = True
                break                                            # recompute sizes after each merge
        if not merged:
            break
    order = sorted(groups, key=lambda k: (bbox(groups[k])[2], bbox(groups[k])[0]))
    out = np.empty(len(gx), int)
    for n, k in enumerate(order):
        out[np.fromiter(groups[k], int)] = n
    return out


# -------------------------------------------------------------------- slide ----
def process(row: dict, svs: Path, out: Path, ref: dict, panel: list[str]) -> dict:
    import openslide
    t0 = time.time()
    rec = dict(file_id=row["file_id"], file_name=row["file_name"], patient=row["patient"],
               kind=row["kind"], token=slide_token(row["file_name"]), stratum=row.get("stratum"))
    sl = openslide.OpenSlide(str(svs))
    pr = sl.properties
    mpp_x, mpp_y = pr.get("openslide.mpp-x"), pr.get("openslide.mpp-y")
    rec.update(mpp_x=mpp_x, mpp_y=mpp_y, objective=pr.get("openslide.objective-power"),
               vendor=pr.get("openslide.vendor"), scanner=pr.get("aperio.ScanScope ID"),
               level0=list(sl.dimensions), n_levels=sl.level_count)
    if not mpp_x or not mpp_y:
        rec["status"] = "no_mpp"
        return rec
    mpp = (float(mpp_x) + float(mpp_y)) / 2
    if not (0.15 <= mpp <= 1.1) or abs(float(mpp_x) - float(mpp_y)) / mpp > 0.02:
        rec.update(status="bad_mpp", mpp=mpp)
        return rec
    rec["mpp"] = mpp
    W0um, H0um = sl.dimensions[0] * mpp, sl.dimensions[1] * mpp

    # ---- 3 mask ------------------------------------------------------------------------
    mu = P["MASK_UM"]
    mw, mh = int(W0um / mu), int(H0um / mu)
    thumb, _ = read_scaled(sl, 0, 0, mw, mh, mu, mpp, max_level_um=mu * 2)
    mask, pen, s_thr = tissue_mask(thumb)
    rec.update(sat_threshold=s_thr, tissue_mm2=round(mask.sum() * mu * mu / 1e6, 2),
               pen_mm2=round(pen.sum() * mu * mu / 1e6, 3))
    if not mask.any():
        rec["status"] = "no_tissue"
        return rec

    # ---- 4 grid ------------------------------------------------------------------------
    ys, xs = np.nonzero(mask)
    ox, oy = xs.min() * mu, ys.min() * mu                              # tissue bbox corner (um)
    pitch = P["PITCH_UM"]
    nx = int(math.ceil((xs.max() + 1) * mu - ox) / pitch) + 1
    ny = int(math.ceil((ys.max() + 1) * mu - oy) / pitch) + 1
    r = P["SPOT_UM"] / 2 / mu
    k = int(math.ceil(r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = ((xx ** 2 + yy ** 2) <= r * r).astype(np.float32)
    frac_map = cv2.filter2D(mask.astype(np.float32), -1, disk / disk.sum(), borderType=cv2.BORDER_CONSTANT)
    gi, gj = np.meshgrid(np.arange(nx), np.arange(ny))                 # gi = column (x), gj = row (y)
    cx_um = ox + pitch / 2 + gi * pitch
    cy_um = oy + pitch / 2 + gj * pitch
    px_m = np.clip((cx_um / mu).astype(int), 0, mw - 1)
    py_m = np.clip((cy_um / mu).astype(int), 0, mh - 1)
    frac = frac_map[py_m, px_m]
    inside = (cx_um < W0um) & (cy_um < H0um)
    keep = (frac >= P["MIN_TISSUE"]) & inside
    spots = pd.DataFrame(dict(gx=gi[keep], gy=gj[keep], cx_um=cx_um[keep], cy_um=cy_um[keep],
                              tissue_frac=frac[keep]))
    rec["n_spots"] = len(spots)
    if spots.empty:
        rec["status"] = "no_spots"
        return rec

    # ---- 5 windows ---------------------------------------------------------------------
    spots["win"] = assign_windows(spots.gx.values.astype(int), spots.gy.values.astype(int))

    # ---- 6 export ----------------------------------------------------------------------
    um_ref = ref["um_per_px_ref"]
    ppu = pitch / um_ref                                               # her2st px per array unit
    keep_px = P["KEEP_PX"]
    data = out / "data"
    for d in ("ST-imgs", "ST-spotfiles", "ST-cnts"):
        (data / d).mkdir(parents=True, exist_ok=True)
    prefix = f"{rec['patient']}_{rec['token']}_w"                   # drop a previous run's windows
    for old in list((data / "ST-imgs" / rec["patient"]).glob(prefix + "*")) if (data / "ST-imgs" / rec["patient"]).exists() else []:
        for f in old.glob("*"):
            f.unlink()
        old.rmdir()
    for old in list((data / "ST-spotfiles").glob(prefix + "*")) + list((data / "ST-cnts").glob(prefix + "*")):
        old.unlink()
    windows = []
    for w, g in spots.groupby("win", sort=True):
        wn = int(w) + 1
        sec = f"{rec['patient']}_{rec['token']}_w{wn:02d}"
        gx0, gy0 = int(g.gx.min()), int(g.gy.min())
        # image box: spot centres' bbox +- keep_px (her2st px), at her2st scale
        x0_um = g.cx_um.min() - keep_px * um_ref
        y0_um = g.cy_um.min() - keep_px * um_ref
        w_out = int(math.ceil((g.cx_um.max() - g.cx_um.min()) / um_ref)) + 2 * keep_px + 1
        h_out = int(math.ceil((g.cy_um.max() - g.cy_um.min()) / um_ref)) + 2 * keep_px + 1
        img, lv = read_scaled(sl, x0_um, y0_um, w_out, h_out, um_ref, mpp, strip=P["STRIP_OUT_PX"])
        px = (g.cx_um.values - x0_um) / um_ref
        py = (g.cy_um.values - y0_um) / um_ref
        keepm = np.zeros((h_out, w_out), bool)
        for a, b in zip(np.round(px).astype(int), np.round(py).astype(int)):
            keepm[max(0, b - keep_px):b + keep_px + 1, max(0, a - keep_px):a + keep_px + 1] = True
        img[~keepm] = 255
        idir = data / "ST-imgs" / rec["patient"] / sec
        idir.mkdir(parents=True, exist_ok=True)
        for old in idir.glob("*"):
            old.unlink()
        jpg = idir / f"HE_{sec}.jpg"
        cv2.imwrite(str(jpg), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, P["JPEG_Q"]])
        del img, keepm
        lx = (g.gx.values - gx0 + 1).astype(int)
        ly = (g.gy.values - gy0 + 1).astype(int)
        assert lx.max() <= P["MAX_WIN"] and ly.max() <= P["MAX_WIN"] and lx.min() >= 1 and ly.min() >= 1
        sp = pd.DataFrame(dict(x=lx, y=ly, new_x=lx.astype(float), new_y=ly.astype(float), pixel_x=px,
                               pixel_y=py, selected=1, tissue_frac=g.tissue_frac.values, gx=g.gx.values,
                               gy=g.gy.values, cx_um=g.cx_um.values, cy_um=g.cy_um.values))
        sp.to_csv(data / "ST-spotfiles" / f"{sec}_selection.tsv", sep="\t", index=False, float_format="%.4f")
        ids = [f"{a}x{b}" for a, b in zip(lx, ly)]
        pd.DataFrame(1, index=ids, columns=panel).to_csv(data / "ST-cnts" / f"{sec}.tsv.gz", sep="\t",
                                                         compression="gzip")
        windows.append(dict(section=sec, window=wn, extent_x=int(lx.max()), extent_y=int(ly.max()), n_spots=len(g),
                            img_w=w_out, img_h=h_out, jpg_mb=round(jpg.stat().st_size / 1e6, 1), level=lv))
    rec["windows"] = windows
    rec["n_windows"] = len(windows)
    rec["jpg_mb"] = round(sum(w["jpg_mb"] for w in windows), 1)

    # ---- 7 QC thumbnail ----------------------------------------------------------------
    qs = 1600 / max(mw, mh)
    qc = cv2.resize(thumb, (int(mw * qs), int(mh * qs)), interpolation=cv2.INTER_AREA)
    cs, _ = cv2.findContours(cv2.resize(mask.astype(np.uint8), qc.shape[1::-1], interpolation=cv2.INTER_NEAREST),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(qc, cs, -1, (0, 160, 0), 1)
    penr = cv2.resize(pen.astype(np.uint8), qc.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    qc[penr] = (255, 0, 255)
    rng = np.random.default_rng(0)
    cols = {w: tuple(int(c) for c in rng.integers(0, 200, 3)) for w in range(len(windows) + 1)}
    f = qs / mu
    for w, g in spots.groupby("win"):
        c = cols[int(w)]
        for a, b in zip(g.cx_um.values * f, g.cy_um.values * f):
            rr = max(2, int(P["SPOT_UM"] / 2 * f))
            cv2.circle(qc, (int(a), int(b)), rr + 1, (0, 0, 0), -1)               # dark rim: visible on any stain
            cv2.circle(qc, (int(a), int(b)), rr, c, -1)
        x0, x1 = (g.cx_um.min() - pitch / 2) * f, (g.cx_um.max() + pitch / 2) * f
        y0, y1 = (g.cy_um.min() - pitch / 2) * f, (g.cy_um.max() + pitch / 2) * f
        cv2.rectangle(qc, (int(x0), int(y0)), (int(x1), int(y1)), c, 2)
        cv2.putText(qc, f"w{int(w) + 1:02d}", (int(x0) + 3, int(y0) + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, c, 1, cv2.LINE_AA)
    (out / "qc").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "qc" / f"{rec['patient']}_{rec['token']}.jpg"), cv2.cvtColor(qc, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    rec.update(status="ok", seconds=round(time.time() - t0))
    return rec


def run_one(row: dict, out: Path, work: Path, ref: dict, panel: list[str], local: Path | None = None) -> dict:
    done = out / "meta" / "slides" / f"{row['file_id']}.json"
    if done.exists():
        return json.loads(done.read_text())
    svs = local or work / row["file_name"]
    t0 = time.time()
    try:
        if local is None:
            download(row["file_id"], row["md5sum"], svs)
        t_dl = round(time.time() - t0)
        rec = process(row, svs, out, ref, panel)
        rec["download_s"] = t_dl
    except Exception as e:                                           # noqa: BLE001
        rec = dict(file_id=row["file_id"], file_name=row["file_name"], patient=row["patient"],
                   kind=row["kind"], status="error", error=repr(e), trace=traceback.format_exc()[-2000:])
    finally:
        if local is None:
            svs.unlink(missing_ok=True)
    if rec["status"] != "error":                                      # errors are retried on rerun
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: rec.get(k) for k in ("patient", "token", "status", "mpp", "tissue_mm2", "n_spots",
                                              "n_windows", "jpg_mb", "download_s", "seconds", "error")},
                     default=float), flush=True)
    return rec


def collect(out: Path) -> None:
    recs = [json.loads(p.read_text()) for p in sorted((out / "meta" / "slides").glob("*.json"))]
    sl = pd.DataFrame([{k: v for k, v in r.items() if k != "windows"} for r in recs])
    sl.to_csv(out / "meta" / "slides.tsv", sep="\t", index=False)
    win = pd.DataFrame([dict(w, file_id=r["file_id"], patient=r["patient"], kind=r["kind"], token=r["token"],
                             stratum=r.get("stratum")) for r in recs for w in r.get("windows", [])])
    win.to_csv(out / "meta" / "windows.tsv", sep="\t", index=False)
    print(sl.status.value_counts().to_string())
    if len(win):
        print(f"{len(win)} windows, {int(win.n_spots.sum())} spots, {win.jpg_mb.sum() / 1e3:.1f} GB JPEG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCI / "tcga_upload"))
    ap.add_argument("--work", default=str(SCI / "tcga_work" / "svs"))
    ap.add_argument("--cohort", default=str(REPO / "results" / "tcga_cohort"))
    ap.add_argument("--scale", default=str(SCI / "ext_results" / "calib" / "her2st_scale.json"))
    ap.add_argument("--panel", default=str(REPO / "panels" / "panel_833.txt"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N slides (testing)")
    ap.add_argument("--patients", nargs="*", help="only these patients (testing)")
    ap.add_argument("--kinds", nargs="*", help="only these slide kinds, e.g. DX (testing)")
    ap.add_argument("--local", nargs="*", help="process these .svs files already on disk (no download)")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    if a.collect:
        collect(out)
        return
    ref = json.loads(Path(a.scale).read_text())
    panel = [ln.split("\t")[0].strip() for ln in Path(a.panel).read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "params.json").write_text(json.dumps(dict(P, her2st_scale=ref, panel_n=len(panel)), indent=1))
    work.mkdir(parents=True, exist_ok=True)
    slides = pd.read_csv(Path(a.cohort) / "slides.csv")
    slides = slides[slides.role == "main"]
    if a.local is not None:
        by_name = slides.set_index("file_name")
        for f in a.local:
            row = dict(by_name.loc[Path(f).name], file_name=Path(f).name)
            run_one(row, out, work, ref, panel, local=Path(f))
        return
    if a.patients:
        slides = slides[slides.patient.isin(a.patients)]
    if a.kinds:
        slides = slides[slides.kind.isin(a.kinds)]
    rows = slides.to_dict("records")
    rows.sort(key=lambda r: -r["file_size"])                          # big first: better packing
    if a.limit:
        rows = rows[:a.limit]
    from multiprocessing import Pool
    with Pool(a.jobs, maxtasksperchild=1) as pool:
        pool.starmap(run_one, [(r, out, work, ref, panel) for r in rows], chunksize=1)
    collect(out)


if __name__ == "__main__":
    main()
