#!/usr/bin/env python
"""Stage 0.3 -- recreate DeepPT's LAST-epoch MLP weights.

Why: 03_run_lopo.py saves <P>_mlp.pt only when validation PCC improves, i.e. the
BEST-VAL epoch, but the main table scores DeepPT at the LAST (early-stopped) epoch.
The last-epoch weights were never written, so the Visium arm cannot use the stored
checkpoint without switching epoch rules.

What this does: runs the port's OWN 03_run_lopo.py source, unchanged except for one
inserted line that also saves the final MLP as <P>_mlp_last.pt, into a NEW output dir
(the original run is never touched).  Same features, targets, config and seed 42 with
cudnn deterministic, so the rerun should reproduce the original fold by fold.  At the
end it compares every fold's last-epoch predictions with the original run's.

usage: cd /workspace/st_benching && python external/deeppt_resave_last.py
       (~30-45 min for 8 folds; the AE is refit per fold for 500 epochs, as in the run)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = "/workspace/st_benching/deeppt_her2st"
ORIG = "/workspace/deeppt/results/deeppt_833_raw"


def per_gene_pcc(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="/workspace/deeppt/features_raw")
    ap.add_argument("--targets", default="/workspace/deeppt/targets")
    ap.add_argument("--out", default="/workspace/deeppt/results/deeppt_833_raw_lastckpt")
    ap.add_argument("--folds", nargs="*", default=list("ABCDEFGH"))
    ap.add_argument("--compare_only", action="store_true")
    a = ap.parse_args()

    if not a.compare_only:
        path = os.path.join(SRC_DIR, "03_run_lopo.py")
        src = open(path).read()
        anchor = '    pd.DataFrame(hist).to_csv(os.path.join(pdir, "history.csv"), index=False)\n'
        assert src.count(anchor) == 1, "03_run_lopo.py changed -- anchor line not found exactly once"
        src = src.replace(anchor, anchor + '    torch.save(mlp.state_dict(), '
                          'os.path.join(args.out, "ckpt", f"{P}_mlp_last.pt"))\n')
        # same CLI the original run used (run_all.sh), defaults for everything else
        sys.argv = ["03_run_lopo.py", "--features", a.features, "--targets", a.targets,
                    "--out", a.out, "--tag", "raw", "--folds", *a.folds]
        sys.path.insert(0, SRC_DIR)
        exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})

    rows = []
    for P in a.folds:
        h_new = pd.read_csv(os.path.join(a.out, "preds", P, "history.csv"))
        h_old = pd.read_csv(os.path.join(ORIG, "preds", P, "history.csv"))
        e_new, e_old = int(h_new.epoch.iloc[-1]), int(h_old.epoch.iloc[-1])
        n = np.load(os.path.join(a.out, "preds", P, f"{P}_{e_new}.npz"), allow_pickle=True)
        o = np.load(os.path.join(ORIG, "preds", P, f"{P}_{e_old}.npz"), allow_pickle=True)
        same_rows = list(n["spot_id"]) == list(o["spot_id"]) and list(n["section"]) == list(o["section"])
        d = float(np.abs(n["counts"] - o["counts"]).max()) if same_rows and e_new == e_old else np.nan
        # PCC of new vs old predictions per gene: 1.0 = identical up to scale
        r = float(np.nanmean(per_gene_pcc(n["counts"], o["counts"]))) if same_rows else np.nan
        rows.append(dict(fold=P, last_epoch_orig=e_old, last_epoch_new=e_new,
                         rows_identical=same_rows, max_abs_pred_diff=d, mean_pcc_new_vs_old=round(r, 6),
                         mlp_last_saved=os.path.exists(os.path.join(a.out, "ckpt", f"{P}_mlp_last.pt"))))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(a.out, "resave_vs_original.csv"), index=False)
    ok = (df.last_epoch_orig == df.last_epoch_new).all() and (df.max_abs_pred_diff < 1e-4).all()
    print("REPRODUCED EXACTLY" if ok else
          "NOT BIT-EXACT: see table -- if epochs match and new-vs-old PCC > 0.999 the retrain is "
          "equivalent (GPU nondeterminism); if last epochs differ, early stopping diverged -> tell Claude")


if __name__ == "__main__":
    main()
