"""
Score the within-patient run on the benchmark's own footing: per-fold
per-gene PCC, averaged across folds. Never pool spots across folds before
computing PCC -- that is the section-level batch-effect artefact that
inverted signs earlier in this project.

    python score_bleep.py --run_dir /workspace/runs/bleep_patientB \
        --out_dir ../results/bleep_patientB_833
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

MARKERS = ["ERBB2", "GRB7", "ESR1", "PGR", "FASN", "GNAS", "MKI67"]


def per_gene_pcc(pred, truth):
    p = pred - pred.mean(axis=0, keepdims=True)
    t = truth - truth.mean(axis=0, keepdims=True)
    denom = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (p * t).sum(axis=0) / denom
    r[denom == 0] = np.nan
    return r


def per_spot_pcc(pred, truth):
    return per_gene_pcc(pred.T, truth.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fold_files = sorted(glob.glob(os.path.join(args.run_dir, "*", "preds.npz")))
    if not fold_files:
        raise SystemExit(f"no preds.npz under {args.run_dir}/*/")

    gene_rows, spot_rows, genes = {}, {}, None
    for f in fold_files:
        fold = os.path.basename(os.path.dirname(f))
        z = np.load(f, allow_pickle=True)
        pred, truth, base = z["pred"], z["truth"], z["baseline"]
        if genes is None:
            genes = list(z["genes"])
        elif list(z["genes"]) != genes:
            raise SystemExit(f"{fold}: gene order differs from earlier folds")

        gene_rows[fold] = per_gene_pcc(pred, truth)
        sp = per_spot_pcc(pred, truth)

        # gain over the per-gene mean-expression baseline (SSE ratio < 1 = better)
        sse_model = ((pred - truth) ** 2).sum()
        sse_base = ((base[None, :] - truth) ** 2).sum()
        spot_rows[fold] = {
            "n_spots": pred.shape[0],
            "per_gene_pcc_median": np.nanmedian(gene_rows[fold]),
            "per_gene_pcc_mean": np.nanmean(gene_rows[fold]),
            "pct_genes_positive": 100 * np.nanmean(gene_rows[fold] > 0),
            "per_spot_pcc_median": np.nanmedian(sp),
            "sse_ratio_vs_mean_baseline": sse_model / sse_base,
        }

    gene_df = pd.DataFrame(gene_rows, index=genes)
    gene_df["mean_across_folds"] = gene_df.mean(axis=1, skipna=True)
    gene_df.to_csv(os.path.join(args.out_dir, "per_gene_pcc_by_fold.csv"))

    fold_df = pd.DataFrame(spot_rows).T
    fold_df.index.name = "fold"
    fold_df.to_csv(os.path.join(args.out_dir, "per_fold_summary.csv"))

    headline = {
        "n_folds": len(fold_files),
        "folds": list(gene_rows),
        "per_gene_pcc_mean_across_folds":
            float(fold_df["per_gene_pcc_mean"].mean()),
        "per_gene_pcc_median_across_folds":
            float(fold_df["per_gene_pcc_median"].median()),
        "fold_range": [float(fold_df["per_gene_pcc_mean"].min()),
                       float(fold_df["per_gene_pcc_mean"].max())],
        "per_spot_pcc_median_across_folds":
            float(fold_df["per_spot_pcc_median"].median()),
        "sse_ratio_vs_mean_baseline_mean":
            float(fold_df["sse_ratio_vs_mean_baseline"].mean()),
        "markers": {m: float(gene_df.loc[m, "mean_across_folds"])
                    for m in MARKERS if m in gene_df.index},
    }
    with open(os.path.join(args.out_dir, "headline.json"), "w") as fh:
        json.dump(headline, fh, indent=2)

    print(fold_df.to_string())
    print()
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
