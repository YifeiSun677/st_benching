"""
ridge/stnet_denominator.py -- recover ST-Net's per-spot denominator so the ridge
baseline can be fitted on ST-Net's own target scale (data target 'stnet_lognorm').

ST-Net (gene_transform=log, norm=norm) stores truth as ln((1 + C) / d_spot), where d_spot
is a per-spot total over ST-Net's own gene list, which this repo does not have. Since
ln(1 + C) - truth is the same for every gene of a spot, d_spot can be read back from
the stored truth exactly. Run once wherever ST-Net's prediction files and the raw
counts both exist, then copy the CSV to config.STNET_DENOM:

    python -m ridge.stnet_denominator --stnet-dir <dir with A_25.npz ... H_25.npz> \
        --her2st-root <her2st/data> --out panels/stnet_denominator.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config, data


def recover(stnet_dir, her2st_root, epoch=25, patients=config.PATIENTS, genes=None):
    genes = genes or data.load_panel()
    rows, worst = [], 0.0
    for P in patients:
        z = np.load(f"{stnet_dir}/{P}_{epoch}.npz", allow_pickle=True)
        assert [str(g) for g in z["gene_names"]] == genes, f"{P}: ST-Net gene order differs from the panel"
        secs = np.array([f"{P}{s}" for s in z["section"]])
        sids = np.array([f"{int(x)}x{int(y)}" for x, y in z["coord"]])
        T = z["counts"].astype(np.float64)
        for sec in dict.fromkeys(secs):
            m = secs == sec
            C = data.read_counts_df(sec, her2st_root).reindex(columns=genes, fill_value=0).loc[sids[m]].values
            D = np.log1p(C.astype(np.float64)) - T[m]                  # = ln d_spot, for every gene
            worst = max(worst, float((D.max(1) - D.min(1)).max()))
            rows.append(pd.DataFrame(dict(key=[f"{sec}:{s}" for s in sids[m]], denominator=np.exp(np.median(D, 1)))))
    out = pd.concat(rows, ignore_index=True)
    assert out.key.is_unique, "duplicate spot keys"
    assert worst < 1e-4, f"ln(1+C) - truth is not constant within a spot (spread {worst:.2e}); not ST-Net's form"
    return out, worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stnet-dir", required=True)
    ap.add_argument("--her2st-root", default=config.HER2ST_ROOT)
    ap.add_argument("--epoch", type=int, default=25)
    ap.add_argument("--out", default=config.STNET_DENOM)
    args = ap.parse_args(argv)
    out, worst = recover(args.stnet_dir, args.her2st_root, args.epoch)
    out.to_csv(args.out, index=False, float_format="%.10g")
    print(f"[stnet_denominator] {len(out)} spots, within-spot spread {worst:.1e} -> {args.out}")


if __name__ == "__main__":
    main()
