#!/usr/bin/env python
"""
Step 4 — per-fold per-gene PCC, averaged across folds. NEVER pooled.

Pooling predictions across held-out patients on her2st is contaminated by
section-level batch effects and can invert the sign of the real signal. This
script computes PCC inside each fold, then averages the 8 fold values per gene.

Also emits the same fields as your ST-Net / HisToGene tables so the row drops
straight into the comparison:
    median / mean per-gene PCC, % genes positive, marker-gene PCCs,
    per-spot PCC, and gain over the per-fold mean-expression baseline.

Usage:
    python 04_score.py --preds .../results/deeppt_833/preds \
        --targets .../targets --features .../features_raw \
        --out .../results/deeppt_833 --epoch best
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

PATIENTS = list("ABCDEFGH")
MARKERS = ["ERBB2", "GRB7", "ESR1", "FASN", "GNAS", "PGR", "MKI67"]


def corr_cols(A, B):
    """Column-wise Pearson r between two [n, g] matrices; NaN where constant."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    num = (A * B).sum(0)
    den = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0))
    out = np.full(A.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def corr_rows(A, B):
    return corr_cols(A.T, B.T)


def truth_for(patient, targ_dir, feat_dir):
    secs = sorted(s[:-4] for s in os.listdir(targ_dir)
                  if s.endswith(".npy") and s.startswith(patient))
    ys, ids, sections = [], [], []
    for s in secs:
        ys.append(np.load(os.path.join(targ_dir, f"{s}.npy")))
        sp = pd.read_csv(os.path.join(feat_dir, f"{s}_spots.csv"))
        ids += sp["spot_id"].astype(str).tolist()
        sections += [s] * len(sp)
    return np.concatenate(ys), np.array(ids), np.array(sections)


def pick_epoch(pdir, patient, mode):
    if mode != "best":
        return int(mode)
    h = pd.read_csv(os.path.join(pdir, patient, "history.csv"))
    col = "val_gene_pcc" if "val_gene_pcc" in h.columns else "val_mse"
    idx = h[col].idxmax() if col == "val_gene_pcc" else h[col].idxmin()
    return int(h.loc[idx, "epoch"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epoch", default="best",
                    help="'best' (per-fold val minimum) or a fixed integer")
    args = ap.parse_args()

    genes = [l.strip() for l in open(os.path.join(args.targets, "genes.txt")) if l.strip()]
    per_gene, per_fold, baseline = {}, [], {}

    for P in PATIENTS:
        if not os.path.isdir(os.path.join(args.preds, P)):
            continue
        ep = pick_epoch(args.preds, P, args.epoch)
        z = np.load(os.path.join(args.preds, P, f"{P}_{ep}.npz"), allow_pickle=True)
        pred = z["counts"]
        true, ids, secs = truth_for(P, args.targets, args.features)
        assert len(pred) == len(true), f"{P}: {len(pred)} vs {len(true)}"
        assert list(z["spot_id"].astype(str)) == list(ids), f"{P}: spot order mismatch"

        r_gene = corr_cols(pred, true)
        r_spot = corr_rows(pred, true)
        per_gene[P] = r_gene
        baseline[P] = ((true - true.mean(0, keepdims=True)) ** 2).mean() / \
                      ((true - pred) ** 2).mean()

        per_fold.append({
            "patient": P, "epoch": ep, "n_spots": len(true),
            "gene_pcc_median": np.nanmedian(r_gene),
            "gene_pcc_mean": np.nanmean(r_gene),
            "pct_genes_positive": 100 * np.nanmean(r_gene > 0),
            "spot_pcc_median": np.nanmedian(r_spot),
            "mse": float(((true - pred) ** 2).mean()),
            "gain_over_mean_baseline": baseline[P],
        })

    G = pd.DataFrame(per_gene, index=genes)
    G["pcc_mean_across_folds"] = G[list(per_gene)].mean(axis=1)
    G["pcc_sd_across_folds"] = G[list(per_gene)].std(axis=1)
    G = G.sort_values("pcc_mean_across_folds", ascending=False)

    os.makedirs(args.out, exist_ok=True)
    G.to_csv(os.path.join(args.out, "per_gene_pcc.csv"))
    F = pd.DataFrame(per_fold)
    F.to_csv(os.path.join(args.out, "per_fold.csv"), index=False)

    m = G["pcc_mean_across_folds"]
    print(F.to_string(index=False))
    print(f"\n=== DeepPT, 833 panel, LOPO, per-fold per-gene PCC ===")
    print(f"  median {m.median():+.4f}   mean {m.mean():+.4f}   "
          f"positive {100*(m>0).mean():.1f}%")
    print("  markers:", "  ".join(
        f"{g} {m[g]:+.3f}" for g in MARKERS if g in m.index))
    print("\n  (compare: ST-Net median +0.098 / HisToGene median +0.061)")


if __name__ == "__main__":
    main()
