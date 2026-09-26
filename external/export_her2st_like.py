#!/usr/bin/env python
"""Stage 2.2-2.6 -- write one Visium section in exactly the her2st file layout,
at her2st's image scale, under /workspace/ext/visium_like/data (NOT the her2st tree).

  ST-imgs/<P>/<SEC>/HE_<SEC>.jpg          full-res H&E resized by f = um/px_visium / um/px_her2st
  ST-spotfiles/<SEC>_selection.tsv        x, y           = Visium array_col, array_row (unique ints
                                                           -> the 'XxY' ids every port builds stay unique)
                                          new_x, new_y   = her2st-equivalent physical position (200 um units)
                                          pixel_x/_y     = spot centre in the resized image (x = column)
                                          x_eq, y_eq, x_int, y_int, barcode, array_row, array_col (extras)
  ST-cnts/<SEC>.tsv.gz                    raw counts, spots x genes, index 'XxY', genes with >0 counts only
                                          (same convention as her2st: sections carry only detected genes)
  calib/counts_<SEC>.npz                  sparse raw counts, ALL features, same row order (for scorer/Moran)
  calib/panel_coverage.tsv                section, gene, measured (in feature list), detected (>0 counts)
  calib/visium_scale.tsv                  scale bookkeeping

usage: python external/export_her2st_like.py I1 I2 J1 K1
needs: tifffile, opencv-python, h5py, scipy   (pip install tifffile imagecodecs if the tif is compressed)
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common as K


def load_rgb(path):
    import tifffile
    try:
        img = tifffile.imread(path, level=0)        # pyramidal tif: take full resolution
    except TypeError:
        img = tifffile.imread(path)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3]
    if img.dtype != np.uint8:
        hi = np.percentile(img, 99.9)
        img = np.clip(img.astype(np.float32) / max(hi, 1) * 255, 0, 255).astype(np.uint8)
        print("  NOTE: image was not uint8; rescaled by 99.9th percentile")
    return img


def her2st_gene_namespace():
    sec = K.her2st_sections()[0]
    cols = K.read_counts(sec, K.HER2ST_ROOT).columns[:200]
    return "ensg" if np.mean([c.startswith("ENSG") for c in cols]) > 0.5 else "symbol"


def export(sec, ref, panel, ns, args):
    t0 = time.time()
    f = K.find_raw_files(sec)
    pos = K.read_positions(f["positions"])
    sf = K.read_scalefactors(f["scalefactors"])
    X, bc, names, ids = K.read_10x_h5(f["h5"])

    # ---- scale ------------------------------------------------------------
    tis = pos[pos.in_tissue == 1].copy()
    sc = K.visium_scale(tis, sf)
    umpp = sc["um_per_px"]                      # from the 100-um pitch, not spot_diameter_fullres
    ok = sc["hex_skew_pct"] <= 2 and 60 <= sc["implied_spot_diam_um"] <= 70
    if not ok and not args.force:
        raise SystemExit(f"{sec}: scale sanity failed {sc} -- fix Stage 1.")
    fct = umpp / ref["um_per_px_ref"]

    # ---- spots: in tissue, >0 UMI --------------------------------------------
    row_of = {b: i for i, b in enumerate(bc)}
    tis = tis[tis.barcode.isin(list(row_of))]
    Xs = X[[row_of[b] for b in tis.barcode]]
    umi = np.asarray(Xs.sum(1)).ravel()
    keep = umi > 0
    tis, Xs = tis[keep].reset_index(drop=True), Xs[keep]
    n_drop0 = int((~keep).sum())

    tis["pixel_x"] = tis.pxl_col * fct            # image column
    tis["pixel_y"] = tis.pxl_row * fct            # image row
    x_eq = tis.pixel_x / ref["px_per_unit_ref"]
    y_eq = tis.pixel_y / ref["px_per_unit_ref"]
    tis["x_eq"] = x_eq - x_eq.min() + 2           # her2st convention: coords start near 2
    tis["y_eq"] = y_eq - y_eq.min() + 2
    tis["x_int"] = np.round(tis.x_eq).astype(int)
    tis["y_int"] = np.round(tis.y_eq).astype(int)
    mx = int(max(tis.x_int.max(), tis.y_int.max()))
    if mx >= 64 and not args.allow_overflow:
        raise SystemExit(f"{sec}: max integer position {mx} >= 64 (HisToGene/Hist2ST n_pos). "
                         "Tile the section or rerun with --allow_overflow and handle it in the driver.")
    ids_xy = [f"{c}x{r}" for c, r in zip(tis.array_col, tis.array_row)]
    assert len(set(ids_xy)) == len(ids_xy)

    # ---- image --------------------------------------------------------------
    import cv2
    img = load_rgb(f["image"])
    H, W = img.shape[:2]
    newW, newH = int(round(W * fct)), int(round(H * fct))
    interp = cv2.INTER_AREA if fct < 1 else cv2.INTER_LANCZOS4
    small = cv2.resize(img, (newW, newH), interpolation=interp)
    del img
    idir = K.VIS_ROOT / "ST-imgs" / sec[0] / sec
    idir.mkdir(parents=True, exist_ok=True)
    for old in idir.glob("*"):
        old.unlink()                                # ports read the FIRST file in this dir
    cv2.imwrite(str(idir / f"HE_{sec}.jpg"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert tis.pixel_x.max() < newW and tis.pixel_y.max() < newH

    # ---- spot file ------------------------------------------------------------
    spot = pd.DataFrame({
        "x": tis.array_col.values, "y": tis.array_row.values,
        "new_x": tis.x_eq.values, "new_y": tis.y_eq.values,
        "pixel_x": tis.pixel_x.values, "pixel_y": tis.pixel_y.values, "selected": 1,
        "x_eq": tis.x_eq.values, "y_eq": tis.y_eq.values,
        "x_int": tis.x_int.values, "y_int": tis.y_int.values,
        "barcode": tis.barcode.values, "array_row": tis.array_row.values,
        "array_col": tis.array_col.values})
    (K.VIS_ROOT / "ST-spotfiles").mkdir(parents=True, exist_ok=True)
    spot.to_csv(K.VIS_ROOT / "ST-spotfiles" / f"{sec}_selection.tsv", sep="\t", index=False,
                float_format="%.4f")

    # ---- counts ----------------------------------------------------------------
    gnames = ids if ns == "ensg" else names
    np.savez_compressed(K.CALIB / f"counts_{sec}.npz", data=Xs.data, indices=Xs.indices,
                        indptr=Xs.indptr, shape=Xs.shape, genes=np.array(gnames),
                        gene_ids=np.array(ids), spot_id=np.array(ids_xy))
    tot = np.asarray(Xs.sum(0)).ravel()
    det = tot > 0
    dense = pd.DataFrame(Xs[:, det].toarray().astype(np.int32), index=ids_xy,
                         columns=np.array(gnames)[det])
    n_dup = int(dense.columns.duplicated().sum())
    if n_dup:
        dense = dense.T.groupby(level=0).sum().T         # duplicate symbols -> summed
    (K.VIS_ROOT / "ST-cnts").mkdir(parents=True, exist_ok=True)
    dense.to_csv(K.VIS_ROOT / "ST-cnts" / f"{sec}.tsv.gz", sep="\t", compression="gzip")

    # ---- coverage ---------------------------------------------------------------
    feat = set(gnames)
    detected = set(dense.columns)
    cov = pd.DataFrame({"section": sec, "gene": panel,
                        "measured": [int(g in feat) for g in panel],
                        "detected": [int(g in detected) for g in panel]})
    scale = dict(section=sec, um_per_px_visium=umpp, hex_skew_pct=sc["hex_skew_pct"],
                 implied_spot_diam_um=sc["implied_spot_diam_um"], um_per_px_her2st=ref["um_per_px_ref"], f=fct,
                 img_in=f"{W}x{H}", img_out=f"{newW}x{newH}", n_spots=len(spot),
                 n_dropped_zero_umi=n_drop0, max_int_pos=mx, n_dup_symbols=n_dup,
                 panel_measured=int(cov.measured.sum()), panel_detected=int(cov.detected.sum()),
                 genes_written=dense.shape[1], seconds=round(time.time() - t0))
    print(json.dumps(scale, default=float))
    return cov, scale


def upsert(path, df, key="section"):
    if path.exists():
        old = pd.read_csv(path, sep="\t")
        df = pd.concat([old[~old[key].isin(df[key].unique())], df], ignore_index=True)
    df.to_csv(path, sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=K.VISIUM_SECTIONS)
    ap.add_argument("--force", action="store_true", help="ignore the scale sanity checks")
    ap.add_argument("--allow_overflow", action="store_true", help="allow integer positions >= 64")
    a = ap.parse_args()
    K.CALIB.mkdir(parents=True, exist_ok=True)
    ref = json.loads((K.CALIB / "her2st_scale.json").read_text())
    panel = K.load_panel()
    ns = her2st_gene_namespace()
    print(f"her2st gene namespace: {ns};  um/px her2st ref {ref['um_per_px_ref']:.4f}")
    for sec in a.sections:
        assert sec in K.VISIUM_SECTIONS, sec
        cov, scale = export(sec, ref, panel, ns, a)
        upsert(K.CALIB / "panel_coverage.tsv", cov)
        upsert(K.CALIB / "visium_scale.tsv", pd.DataFrame([scale]))


if __name__ == "__main__":
    main()
