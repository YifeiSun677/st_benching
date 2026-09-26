#!/usr/bin/env python
"""Stage 1.1 -- one line per downloaded Visium section; nothing is modified.

usage:  python external/verify_raw.py            (sections I1 I2 J1 K1)
writes: /workspace/ext/calib/raw_summary.tsv
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common as K


def image_shape(path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        s = t.series[0].levels[0].shape if t.series[0].levels else t.pages[0].shape
    s = tuple(int(v) for v in s)
    if len(s) == 3 and s[0] in (3, 4):          # (C, H, W)
        s = (s[1], s[2], s[0])
    return s                                     # (H, W, C)


def main():
    secs = sys.argv[1:] or K.VISIUM_SECTIONS
    rows = []
    for sec in secs:
        f = K.find_raw_files(sec)
        pos = K.read_positions(f["positions"])
        sf = K.read_scalefactors(f["scalefactors"])
        X, bc, names, ids = K.read_10x_h5(f["h5"])
        H, W = image_shape(f["image"])[:2]
        tis = pos[pos.in_tissue == 1]
        umpp = K.VISIUM_SPOT_UM / sf["spot_diameter_fullres"]
        nb_px = K.neighbour_distance_px(tis)
        umpp_nb = K.VISIUM_PITCH_UM / nb_px
        inside = (tis.pxl_row.max() < H) and (tis.pxl_col.max() < W) and (tis[["pxl_row", "pxl_col"]].min().min() >= 0)
        panel = K.load_panel()
        nm = set(names)
        rows.append(dict(
            section=sec, image=f["image"].name, img_h=H, img_w=W,
            spots_in_tissue=len(tis), barcodes_in_h5=len(bc),
            genes=len(names), ids_are_ensg=all(i.startswith("ENSG") for i in ids[:50]),
            panel_genes_in_features=sum(g in nm for g in panel),
            spot_diameter_fullres=round(sf["spot_diameter_fullres"], 3),
            um_per_px_from_spot=round(umpp, 4), um_per_px_from_pitch=round(umpp_nb, 4),
            scale_mismatch_pct=round(100 * abs(umpp - umpp_nb) / umpp_nb, 2),
            spots_inside_image=bool(inside),
            median_umi=float(np.median(np.asarray(X.sum(1)).ravel()))))
    df = pd.DataFrame(rows)
    K.CALIB.mkdir(parents=True, exist_ok=True)
    df.to_csv(K.CALIB / "raw_summary.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.T.to_string())
    bad = df[(df.scale_mismatch_pct > 3) | (~df.spots_inside_image)]
    if len(bad):
        print("\nFAIL:", list(bad.section), "-- wrong image or scalefactors; fix Stage 1 before Stage 2")
        sys.exit(1)
    print("\nPASS: all sections consistent")


if __name__ == "__main__":
    main()
