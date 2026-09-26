#!/usr/bin/env python
"""Stage 2.1 -- her2st reference scale.

For each of the 36 her2st sections fit pixel ~ array coordinate (array units are
200 um apart), per axis.  s_ref = 200 / median(px per unit).  Same arithmetic as
stflow_port/her2st_io.estimate_um_per_px.

writes: /workspace/ext/calib/her2st_scale.tsv, her2st_scale.json
exit 2 if the across-section spread (max/min - 1) exceeds 5 %  (override: --allow_spread)
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common as K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_spread", action="store_true")
    a = ap.parse_args()
    rows = []
    for sec in K.her2st_sections():
        sp = K.read_spots(sec, K.HER2ST_ROOT)
        xcol = "new_x" if "new_x" in sp else "x"
        ycol = "new_y" if "new_y" in sp else "y"
        sp = sp.dropna(subset=["pixel_x", "pixel_y"])
        sx, ix = np.polyfit(sp[xcol], sp["pixel_x"], 1)
        sy, iy = np.polyfit(sp[ycol], sp["pixel_y"], 1)
        rx = np.sqrt(np.mean((sp["pixel_x"] - (sx * sp[xcol] + ix)) ** 2))
        ry = np.sqrt(np.mean((sp["pixel_y"] - (sy * sp[ycol] + iy)) ** 2))
        ppu = (abs(sx) + abs(sy)) / 2
        rows.append(dict(section=sec, n_spots=len(sp), slope_x=sx, slope_y=sy,
                         xy_diff_pct=100 * abs(abs(sx) - abs(sy)) / ppu,
                         rmse_px=(rx + ry) / 2, px_per_unit=ppu,
                         um_per_px=K.LEGACY_PITCH_UM / ppu))
    df = pd.DataFrame(rows)
    ppu_ref = float(np.median(df.px_per_unit))
    spread = float(df.px_per_unit.max() / df.px_per_unit.min() - 1)
    out = dict(px_per_unit_ref=ppu_ref, um_per_px_ref=K.LEGACY_PITCH_UM / ppu_ref,
               spread_pct=100 * spread, n_sections=len(df),
               um_per_px_min=float(df.um_per_px.min()), um_per_px_max=float(df.um_per_px.max()))
    K.CALIB.mkdir(parents=True, exist_ok=True)
    df.to_csv(K.CALIB / "her2st_scale.tsv", sep="\t", index=False, float_format="%.4f")
    (K.CALIB / "her2st_scale.json").write_text(json.dumps(out, indent=2))
    print(df.round(3).to_string(index=False))
    print(json.dumps(out, indent=2))
    if spread > 0.05 and not a.allow_spread:
        print(f"STOP: spread {100*spread:.1f}% > 5% -- models were trained on mixed scales; "
              "decide the reference (median vs per-patient) before exporting.")
        sys.exit(2)


if __name__ == "__main__":
    main()
