#!/usr/bin/env python
"""
Score a STFlow run on the panel-CP10K footing and write one comparable table.

The new run's predictions are ALREADY on that footing, so they are scored with
--renorm none. The old raw-target run can be put on the same footing with
--renorm panel_cp10k, which expm1's both pred and truth back to counts and
renormalises. Running both gives a like-for-like table.

Metrics, matching the benchmark's established footing:
  per_gene_pcc          per-gene PCC within the fold, averaged across genes
  frac_positive         fraction of genes with PCC > 0
  sse_ratio_median      median over genes of SSE(pred) / SSE(gene mean of truth)
  frac_beat_baseline    fraction of genes with sse_ratio < 1
  sd_ratio_median       median over genes of sd(pred) / sd(truth)
Fold means are unweighted across folds. Genes with zero truth variance in a fold
are dropped from the PCC average and counted.

Usage
  python stflow_port/score_norm.py --run /workspace/runs/stflow_lopo_833_normtarget_e90 \
      --renorm none --out /workspace/runs/stflow_lopo_833_normtarget_e90/scored
  python stflow_port/score_norm.py --run /workspace/runs/stflow_lopo_833_hest112_e90 \
      --renorm panel_cp10k --out /workspace/runs/stflow_lopo_833_hest112_e90/scored_cp10k
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stflow_port.norm_target import panel_cp10k_log1p  # noqa: E402

TRUTH_KEYS = ("truth", "true", "target", "y_true", "counts", "y")
PRED_KEYS = ("pred", "predictions", "y_pred", "prediction")
MARKERS = ("ERBB2", "GRB7", "ESR1", "PGR", "FASN", "GNAS", "MKI67")


def _pick(z, cands, what):
    for k in cands:
        if k in z.files:
            return k
    raise KeyError(f"no {what} array in npz; keys are {z.files}")


def discover(run_dir, pattern=None):
    """Return {fold_name: [section npz paths]} for the layout
    run/fold*/preds/<section>.npz, falling back to flatter layouts.
    Directories named _scored* are skipped."""
    if pattern:
        files = sorted(glob.glob(pattern, recursive=True))
    else:
        files = []
        for pat in (os.path.join(run_dir, "*", "preds", "*.npz"),
                    os.path.join(run_dir, "*", "preds*.npz"),
                    os.path.join(run_dir, "*", "*.npz"),
                    os.path.join(run_dir, "*.npz")):
            files = sorted(glob.glob(pat))
            if files:
                break
    files = [f for f in files if os.sep + "_scored" not in f]
    if not files:
        raise SystemExit(f"no npz files found under {run_dir}")
    folds = {}
    for f in files:
        d = os.path.dirname(f)
        fold = (os.path.basename(os.path.dirname(d))
                if os.path.basename(d) == "preds" else os.path.basename(d))
        if fold in ("", os.path.basename(run_dir)):
            fold = os.path.splitext(os.path.basename(f))[0]
        folds.setdefault(fold, []).append(f)
    return {k: sorted(v) for k, v in sorted(folds.items())}


def load_group(paths, want_pred=True):
    """Concatenate one fold's sections. Returns (pred, truth, genes, sections)."""
    P, T, S, genes = [], [], [], None
    for f in paths:
        z = np.load(f, allow_pickle=True)
        t = np.asarray(z[_pick(z, TRUTH_KEYS, "truth")], dtype=np.float64)
        T.append(t)
        if want_pred:
            P.append(np.asarray(z[_pick(z, PRED_KEYS, "pred")], dtype=np.float64))
        if "genes" in z.files:
            g = [str(x) for x in z["genes"]]
            if genes is None:
                genes = g
            elif g != genes:
                raise SystemExit(f"gene order differs in {f}")
        S += [os.path.splitext(os.path.basename(f))[0]] * t.shape[0]
    return (np.concatenate(P, 0) if want_pred else None,
            np.concatenate(T, 0), genes, np.array(S))


def renorm(arr, mode):
    """arr is log1p(raw counts). Return it on the requested footing."""
    if mode == "none":
        return np.asarray(arr, dtype=np.float64)
    if mode == "panel_cp10k":
        counts = np.clip(np.expm1(np.asarray(arr, dtype=np.float64)), 0, None)
        return panel_cp10k_log1p(counts).astype(np.float64)
    raise ValueError(mode)


def fold_metrics(pred, truth, genes=None):
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f"pred {p.shape} and truth {t.shape} disagree")

    tsd = t.std(0)
    ok = tsd > 0
    pc, tc = p - p.mean(0), t - t.mean(0)
    num = (pc * tc).sum(0)
    den = np.sqrt((pc ** 2).sum(0) * (tc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        pcc = np.where(den > 0, num / den, np.nan)

    sse = ((p - t) ** 2).sum(0)
    sse_base = ((t - t.mean(0)) ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(sse_base > 0, sse / sse_base, np.nan)
        sdr = np.where(tsd > 0, p.std(0) / tsd, np.nan)

    m = {
        "n_spots": int(p.shape[0]),
        "n_genes": int(p.shape[1]),
        "n_genes_scored": int(np.isfinite(pcc).sum()),
        "n_genes_zero_var": int((~ok).sum()),
        "pcc_mean": float(np.nanmean(pcc)),
        "pcc_median": float(np.nanmedian(pcc)),
        "frac_positive": float(np.nanmean(pcc > 0)),
        "sse_ratio_median": float(np.nanmedian(ratio)),
        "frac_beat_baseline": float(np.nanmean(ratio < 1)),
        "sd_ratio_median": float(np.nanmedian(sdr)),
    }
    if genes is not None:
        idx = {g: i for i, g in enumerate(genes)}
        for g in MARKERS:
            if g in idx:
                m[f"pcc_{g}"] = float(pcc[idx[g]])
    return m, pcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--renorm", choices=["none", "panel_cp10k"], default="none")
    ap.add_argument("--out", default=None)
    ap.add_argument("--glob", default=None, help="override the npz search pattern")
    ap.add_argument("--group", choices=["fold", "section"], default="fold",
                    help="compute per-gene PCC over a whole patient (fold) or per section")
    args = ap.parse_args()

    folds = discover(args.run, args.glob)
    print(f"{len(folds)} folds, {sum(len(v) for v in folds.values())} section files\n")

    rows, all_pcc = [], []
    for name, paths in folds.items():
        pred_raw, truth_raw, genes, sections = load_group(paths, want_pred=True)
        units = ({name: (pred_raw, truth_raw)} if args.group == "fold" else
                 {f"{name}/{s}": (pred_raw[sections == s], truth_raw[sections == s])
                  for s in sorted(set(sections.tolist()))})
        for unit, (pr, tr) in units.items():
            m, pcc = fold_metrics(renorm(pr, args.renorm), renorm(tr, args.renorm), genes)
            m["fold"] = unit
            m["n_sections"] = len(paths) if args.group == "fold" else 1
            rows.append(m)
            all_pcc.append(pcc)
            print(f"{unit:16s} pcc {m['pcc_mean']:+.4f}  med {m['pcc_median']:+.4f}  "
                  f"frac+ {m['frac_positive']:.3f}  sse {m['sse_ratio_median']:.3f}  "
                  f"beat {m['frac_beat_baseline']:.3f}  sd {m['sd_ratio_median']:.3f}")

    head = {
        "run": args.run, "renorm": args.renorm, "n_folds": len(rows),
        "pcc_mean": float(np.mean([r["pcc_mean"] for r in rows])),
        "pcc_mean_min": float(np.min([r["pcc_mean"] for r in rows])),
        "pcc_mean_max": float(np.max([r["pcc_mean"] for r in rows])),
        "pcc_median_of_folds": float(np.median([r["pcc_median"] for r in rows])),
        "sse_ratio_median_of_folds": float(np.median([r["sse_ratio_median"] for r in rows])),
        "frac_beat_baseline_mean": float(np.mean([r["frac_beat_baseline"] for r in rows])),
        "sd_ratio_median_of_folds": float(np.median([r["sd_ratio_median"] for r in rows])),
    }
    print("\nHEADLINE  pcc_mean {pcc_mean:+.4f} [{pcc_mean_min:+.4f}, {pcc_mean_max:+.4f}]  "
          "sse {sse_ratio_median_of_folds:.3f}  beat {frac_beat_baseline_mean:.3f}  "
          "sd {sd_ratio_median_of_folds:.3f}".format(**head))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        cols = sorted({k for r in rows for k in r})
        with open(os.path.join(args.out, "per_fold.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        with open(os.path.join(args.out, "headline.json"), "w") as fh:
            json.dump({"headline": head, "folds": rows}, fh, indent=2)
        np.savez_compressed(os.path.join(args.out, "per_gene_pcc_by_fold.npz"),
                            pcc=np.vstack(all_pcc),
                            folds=np.array([r["fold"] for r in rows]))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
