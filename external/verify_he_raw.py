#!/usr/bin/env python
"""He Stage 1.1 -- one line per downloaded He section; nothing is modified.

Checks per section: all four files parse, spot ids are integers, pixel ~ array fit is a clean
regular grid (xy slopes within 3 %, residual < 10 % of the pitch), every spot lies inside the
image, <= 5 % of spots lack a count row.  A few spots (typically 1-3 per section) are in the spot
file but absent from both the count matrix and the tumour annotation -- the release dropped them,
ST-Net's own prepare step skips them, and export_he_like.py keeps only spots with counts.

usage:  python external/verify_he_raw.py [SEC ...]          (default: every section in metadata)
writes: /workspace/ext/he/calib/raw_summary.tsv
"""
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import he_common as H

Image.MAX_IMAGE_PIXELS = None


def main():
    secs = sys.argv[1:] or H.he_sections()
    rows = []
    for sec in secs:
        p = H.raw_paths(sec)
        sp = H.read_raw_spots(p["spots"])
        cnt = H.read_raw_counts(p["counts"])
        tum = H.read_tumor(p["tumor"])
        with Image.open(p["image"]) as im:
            W, H_ = im.size
        sc = H.fit_scale(sp)
        both = sp.index.intersection(cnt.index)
        umi = cnt.loc[both].sum(1)
        inside = bool((sp.pixel_x.between(0, W - 1) & sp.pixel_y.between(0, H_ - 1)).all())
        rows.append(dict(section=sec, patient=p["patient"], subtype=p["subtype"], img_w=W, img_h=H_,
                         n_spots=len(sp), n_count_rows=len(cnt), spots_with_counts=len(both),
                         no_counts=len(sp) - len(both),
                         zero_umi=int((umi == 0).sum()), median_umi=float(umi.median()),
                         n_genes=cnt.shape[1], ensg=bool(np.mean([c.startswith("ENSG") for c in cnt.columns]) > 0.9),
                         tumor_labelled=sum(s in tum for s in sp.index),
                         tumor_frac=float(np.mean([tum.get(s) == "tumor" for s in sp.index])),
                         px_per_unit=round(sc["px_per_unit"], 2), um_per_px=round(sc["um_per_px"], 4),
                         xy_diff_pct=round(sc["xy_diff_pct"], 2), rmse_frac_pitch=round(sc["rmse_frac_pitch"], 3),
                         spots_inside_image=inside))
        print(f"{sec}: {rows[-1]['n_spots']} spots, {rows[-1]['um_per_px']} um/px", flush=True)
    df = pd.DataFrame(rows)
    H.HE_CALIB.mkdir(parents=True, exist_ok=True)
    df.to_csv(H.HE_CALIB / "raw_summary.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_rows", 200):
        print(df.to_string(index=False))
    print(f"\nspots without counts (dropped at export): {int(df.no_counts.sum())} of {int(df.n_spots.sum())}, "
          f"max {int(df.no_counts.max())} in one section")
    print(f"um/px across sections: median {df.um_per_px.median():.4f}, "
          f"range [{df.um_per_px.min():.4f}, {df.um_per_px.max():.4f}]")
    bad = df[(df.xy_diff_pct > 3) | (df.rmse_frac_pitch > 0.10) | (~df.spots_inside_image)
             | (df.no_counts > 0.05 * df.n_spots) | (~df.ensg)]
    if len(bad):
        print("\nFAIL:", list(bad.section), "-- inspect before exporting (export skips nothing silently)")
        sys.exit(1)
    print("\nPASS: all sections consistent")


if __name__ == "__main__":
    main()
