"""
stflow_port/score.py -- summarise a finished LOPO run on the benchmark footing.

  python stflow_port/score.py --tag stflow_lopo_833_hest112
  python stflow_port/score.py --tag ... --gene_sets_dir /workspace/st_benching/results/gene_sets

Writes into $RUNS_ROOT/<tag>/_scored/ (these small tables are what go into git under
results/stflow_833/; the preds/*.npz stay out of git and are rsynced to the Mac):
  per_fold_summary.csv       one row per held-out patient (+ train_mean control columns)
  per_gene_pcc_by_fold.csv   genes x patients
  headline.json              mean over folds + range, markers, gene-set means

Pipeline check built in: the zero-information control (each fold's TRAINING patients'
per-gene mean) must have sse_ratio >= 1 and frac_genes_beat_baseline == 0 on every
patient, because the section's own mean is the L2-optimal constant.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from metrics import patient_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gene_sets_dir", default=None,
                    help="dir with gene_set_<name>.txt files (all/hvg/svg/marker)")
    a = ap.parse_args()
    root = C.run_dir(a.tag)
    folds = sorted(glob.glob(os.path.join(root, "fold*_*")))
    if not folds:
        sys.exit(f"no fold directories under {root}")
    out = os.path.join(root, "_scored")
    os.makedirs(out, exist_ok=True)

    rows, pcc_cols, genes = [], {}, None
    for fd in folds:
        files = sorted(glob.glob(os.path.join(fd, "preds", "*.npz")))
        if not files:
            print(f"  {os.path.basename(fd)}: no preds yet, skipped")
            continue
        zs = [np.load(f, allow_pickle=False) for f in files]
        g = [str(x) for x in zs[0]["genes"]]
        genes = genes or g
        assert g == genes, f"{fd}: gene order differs"
        pat = str(zs[0]["patient"])
        preds = [z["pred"].astype(np.float64) for z in zs]
        truths = [z["truth"].astype(np.float64) for z in zs]
        ctrl = [np.broadcast_to(z["train_mean"], z["truth"].shape).astype(np.float64) for z in zs]
        m, r, _ = patient_metrics(preds, truths, genes, C.MARKERS)
        mc, _, _ = patient_metrics(ctrl, truths, genes)
        if mc["sse_ratio_median"] < 1 - 1e-9 or mc["frac_genes_beat_baseline"] > 0:
            print(f"  !! pipeline check failed on {pat}: control beats section mean")
        run = json.load(open(os.path.join(fd, "run.json")))
        rows.append(dict(patient=pat, sections=" ".join(str(z["section"]) for z in zs),
                         epoch=int(zs[0]["epoch"]), **m,
                         ctrl_sse_ratio_median=mc["sse_ratio_median"],
                         ctrl_frac_beat=mc["frac_genes_beat_baseline"],
                         sec_train=run.get("sec_train"), peak_gpu_gb=run.get("peak_gpu_gb")))
        pcc_cols[pat] = r
        print(f"  {pat}: pcc_mean {m['pcc_mean']:+.4f}  median {m['pcc_median']:+.4f}  "
              f"frac_pos {m['frac_pos']:.3f}  sse {m['sse_ratio_median']:.3f} "
              f"(ctrl {mc['sse_ratio_median']:.3f})  beat {m['frac_genes_beat_baseline']:.3f}  "
              f"sd {m['sd_ratio_median']:.3f}  ERBB2 {m.get('pcc_ERBB2', np.nan):+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "per_fold_summary.csv"), index=False)
    pg = pd.DataFrame(pcc_cols, index=genes)
    pg.index.name = "gene"
    pg.to_csv(os.path.join(out, "per_gene_pcc_by_fold.csv"))

    num = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("epoch",)]
    head = dict(tag=a.tag, n_folds=len(df),
                mean_over_folds={c: float(df[c].mean()) for c in num},
                range_over_folds={c: [float(df[c].min()), float(df[c].max())] for c in
                                  ["pcc_mean", "pcc_median", "sse_ratio_median",
                                   "frac_genes_beat_baseline"]},
                per_patient_pcc_mean=dict(zip(df["patient"], df["pcc_mean"].round(4))))
    if a.gene_sets_dir:
        sets = {}
        for f in sorted(glob.glob(os.path.join(a.gene_sets_dir, "gene_set_*.txt"))):
            name = os.path.basename(f)[len("gene_set_"):-4]
            gs = [l.strip() for l in open(f) if l.strip()]
            idx = [i for i, g in enumerate(genes) if g in set(gs)]
            per_fold = np.nanmean(pg.values[idx], axis=0)
            sets[name] = dict(n=len(idx), mean_over_folds=float(np.nanmean(per_fold)))
        head["gene_sets"] = sets
    json.dump(head, open(os.path.join(out, "headline.json"), "w"), indent=2)
    print(f"\nHEADLINE ({len(df)} folds): pcc_mean {df.pcc_mean.mean():+.4f} "
          f"[{df.pcc_mean.min():+.4f}, {df.pcc_mean.max():+.4f}]  "
          f"pcc_median {df.pcc_median.mean():+.4f}  sse {df.sse_ratio_median.mean():.3f}  "
          f"beat {df.frac_genes_beat_baseline.mean():.3f}\n-> {out}")


if __name__ == "__main__":
    main()
