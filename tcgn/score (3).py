"""
Score a completed TCGN run on the benchmark's footing.

Uses the KEY methodological rule from the project: per-section per-gene PCC,
then averaged across folds -- NOT pooled across held-out patients (pooling is
contaminated by section-level batch effects and can invert the sign). Also
reports the cross-model-comparable columns you carry for every model:
  per-gene PCC (mean/median), frac_pos, median SSE ratio vs the section mean,
  frac_genes_beat_baseline, pred_sd/true_sd, and the marker genes.

  python score.py --tag tcgn_lopo_833_e50

Writes into OUT_DIR/<tag>/scored/:
  per_gene_pcc_by_fold.csv   gene x fold PCC
  per_fold_summary.csv       per fold: pcc mean/median, frac_pos, sse, sd ratio
  headline.json              overall + per-marker
Copy per_fold_summary.csv + headline.json into st_benching/results/<tag>/ (git);
leave the raw npz on the volume / rsync to the Mac.
"""
import os
import glob
import json
import argparse
import numpy as np

import config as C


def pcc_cols(pred, truth):
    """Per-gene Pearson r between columns of pred and truth. NaN if a column is
    constant (zero variance) in either array."""
    p = pred - pred.mean(0, keepdims=True)
    t = truth - truth.mean(0, keepdims=True)
    num = (p * t).sum(0)
    den = np.sqrt((p * p).sum(0) * (t * t).sum(0))
    out = np.full(pred.shape[1], np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def sse_ratio(pred, truth):
    """Median over genes of SSE(pred)/SSE(section-mean). 1.0 = constant predictor."""
    base = np.repeat(truth.mean(0, keepdims=True), truth.shape[0], axis=0)
    sse_p = ((pred - truth) ** 2).sum(0)
    sse_b = ((base - truth) ** 2).sum(0)
    ok = sse_b > 0
    r = sse_p[ok] / sse_b[ok]
    frac_beat = float((r < 1.0).mean())
    return float(np.median(r)), frac_beat


def sd_ratio(pred, truth):
    ps = pred.std(0); ts = truth.std(0)
    ok = ts > 0
    return float(np.median(ps[ok] / ts[ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    run_root = os.path.join(C.OUT_DIR, args.tag)
    fold_dirs = sorted(d for d in glob.glob(os.path.join(run_root, "*")) if os.path.isdir(d)
                       and os.path.exists(os.path.join(d, "preds")))
    assert fold_dirs, "no fold dirs with preds/ under %s" % run_root

    panel = None
    per_gene = {}       # fold_name -> per-gene pcc (avg over that fold's sections)
    fold_rows = []
    marker_acc = {g: [] for g in C.MARKERS}

    for fd in fold_dirs:
        fold = os.path.basename(fd)
        sec_files = sorted(glob.glob(os.path.join(fd, "preds", "*.npz")))
        sec_pccs, sse_list, beat_list, sd_list = [], [], [], []
        marker_here = {g: [] for g in C.MARKERS}
        for sf in sec_files:
            z = np.load(sf, allow_pickle=True)
            pred, truth = z["pred"].astype(np.float64), z["truth"].astype(np.float64)
            if panel is None:
                panel = list(z["genes"].astype(str))
                gidx = {g: i for i, g in enumerate(panel)}
            pg = pcc_cols(pred, truth)
            sec_pccs.append(pg)
            sr, beat = sse_ratio(pred, truth)
            sse_list.append(sr); beat_list.append(beat)
            sd_list.append(sd_ratio(pred, truth))
            for g in C.MARKERS:
                if g in gidx and np.isfinite(pg[gidx[g]]):
                    marker_here[g].append(pg[gidx[g]])
        fold_pg = np.nanmean(np.vstack(sec_pccs), axis=0)   # avg over sections in fold
        per_gene[fold] = fold_pg
        for g in C.MARKERS:
            if marker_here[g]:
                marker_acc[g].append(np.mean(marker_here[g]))
        fold_rows.append({
            "fold": fold,
            "pcc_mean": float(np.nanmean(fold_pg)),
            "pcc_median": float(np.nanmedian(fold_pg)),
            "frac_pos": float(np.nanmean(fold_pg > 0)),
            "median_sse_ratio": float(np.mean(sse_list)),
            "frac_beat_baseline": float(np.mean(beat_list)),
            "sd_ratio": float(np.mean(sd_list)),
        })

    scored = os.path.join(run_root, "scored")
    os.makedirs(scored, exist_ok=True)

    # per-gene x fold table
    folds = list(per_gene.keys())
    mat = np.vstack([per_gene[f] for f in folds]).T   # genes x folds
    with open(os.path.join(scored, "per_gene_pcc_by_fold.csv"), "w") as f:
        f.write("gene," + ",".join(folds) + "\n")
        for i, g in enumerate(panel):
            f.write(g + "," + ",".join("%.6f" % v for v in mat[i]) + "\n")

    # per-fold summary
    with open(os.path.join(scored, "per_fold_summary.csv"), "w") as f:
        cols = ["fold", "pcc_mean", "pcc_median", "frac_pos",
                "median_sse_ratio", "frac_beat_baseline", "sd_ratio"]
        f.write(",".join(cols) + "\n")
        for r in fold_rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    # headline: average the per-fold per-gene means across folds
    fold_gene_mean = np.array([r["pcc_mean"] for r in fold_rows])
    headline = {
        "tag": args.tag, "n_folds": len(folds), "folds": folds,
        "pcc_mean_over_folds": float(np.mean([r["pcc_mean"] for r in fold_rows])),
        "pcc_median_over_folds": float(np.mean([r["pcc_median"] for r in fold_rows])),
        "frac_pos_over_folds": float(np.mean([r["frac_pos"] for r in fold_rows])),
        "median_sse_ratio_over_folds": float(np.mean([r["median_sse_ratio"] for r in fold_rows])),
        "frac_beat_baseline_over_folds": float(np.mean([r["frac_beat_baseline"] for r in fold_rows])),
        "sd_ratio_over_folds": float(np.mean([r["sd_ratio"] for r in fold_rows])),
        "markers": {g: (float(np.mean(v)) if v else None) for g, v in marker_acc.items()},
        "per_fold": fold_rows,
    }
    with open(os.path.join(scored, "headline.json"), "w") as f:
        json.dump(headline, f, indent=2)

    print(json.dumps({k: headline[k] for k in
                      ["pcc_mean_over_folds", "pcc_median_over_folds",
                       "frac_pos_over_folds", "median_sse_ratio_over_folds",
                       "frac_beat_baseline_over_folds", "sd_ratio_over_folds",
                       "markers"]}, indent=2))
    print("\nper-fold:")
    for r in fold_rows:
        print("  %-6s pcc_mean=%+.4f  median=%+.4f  frac_pos=%.3f  sse=%.3f  beat=%.3f  sd=%.3f"
              % (r["fold"], r["pcc_mean"], r["pcc_median"], r["frac_pos"],
                 r["median_sse_ratio"], r["frac_beat_baseline"], r["sd_ratio"]))
    print("\nwrote", scored)


if __name__ == "__main__":
    main()
