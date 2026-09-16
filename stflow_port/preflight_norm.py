#!/usr/bin/env python
"""
Preflight for the STFlow normalised-target run. CPU only, seconds to run.

It works off the OLD run's saved predictions, which store truth as log1p(raw
counts). Because that is log1p of integers, expm1 recovers the counts exactly,
so the new target can be rebuilt and checked without touching her2st at all.

Checks, in order:
  1  truth arrays are recoverable integer counts on an 833-column panel
  2  the rebuilt panel_cp10k target matches score.py's --renorm panel_cp10k
     definition (exact, if --score_module is given; otherwise reference-only)
  3  target descriptive stats on both footings, per fold
  4  a Gaussian prior fitted on the pooled non-held-out folds: sd floor hits,
     parameter ranges, and how many genes are dead in training
  5  the depth-only oracle collapses on the new footing (the whole point of the
     retrain) while scoring high on the old one

Usage
  python stflow_port/preflight_norm.py \
      --old_run /workspace/runs/stflow_lopo_833_hest112_e90 \
      --panel panels/panel_833.txt \
      --out /workspace/runs/preflight_normtarget

  # exact parity against your scorer, if score.py exposes a callable
  ... --score_module stflow_port.score:renorm_panel_cp10k
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys

import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stflow_port.norm_target import (  # noqa: E402
    build_target, invert_raw_log1p, panel_cp10k_log1p, target_report)
from stflow_port.gaussian_prior import GaussianPrior  # noqa: E402

TRUTH_KEYS = ("truth", "true", "target", "y_true", "counts", "y")
PRED_KEYS = ("pred", "predictions", "y_pred", "prediction")
MARKERS = ("ERBB2", "GRB7", "ESR1", "PGR", "FASN", "GNAS", "MKI67")


def _pick(z, candidates, what):
    for k in candidates:
        if k in z.files:
            return k
    raise KeyError(f"no {what} array found in npz; keys are {z.files}")


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


def per_gene_pcc(pred, truth):
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    pc = p - p.mean(0, keepdims=True)
    tc = t - t.mean(0, keepdims=True)
    num = (pc * tc).sum(0)
    den = np.sqrt((pc ** 2).sum(0) * (tc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_run", required=True,
                    help="run dir of the completed raw-target LOPO run")
    ap.add_argument("--panel", default=None, help="panel_833.txt, one gene per line")
    ap.add_argument("--out", default=None, help="where to write preflight.json")
    ap.add_argument("--score_module", default=None,
                    help="module:function implementing score.py's panel_cp10k renorm")
    ap.add_argument("--sd_floor", type=float, default=1e-3)
    ap.add_argument("--glob", default=None, help="override the npz search pattern")
    ap.add_argument("--holdout", default=None,
                    help="fold name to treat as held out when fitting the demo prior")
    args = ap.parse_args()

    report = {"old_run": args.old_run, "checks": {}}
    folds = discover(args.old_run, args.glob)
    n_files = sum(len(v) for v in folds.values())
    print(f"[1] {len(folds)} folds, {n_files} section files under {args.old_run}")

    panel = None
    if args.panel:
        panel = [l.strip() for l in open(args.panel) if l.strip()]
        print(f"    panel file: {len(panel)} genes")

    per_fold, counts_by_fold, genes_ref = {}, {}, None
    for name, paths in folds.items():
        _, y_raw, genes, sections = load_group(paths, want_pred=False)
        if genes is not None:
            if genes_ref is None:
                genes_ref = genes
            elif genes != genes_ref:
                raise SystemExit(f"gene order differs in fold {name}")
        counts = invert_raw_log1p(y_raw, round_to_int=True)
        counts_by_fold[name] = counts
        per_fold[name] = {
            "sections": sorted(set(sections.tolist())),
            "raw": target_report(counts, "raw_log1p"),
            "panel_cp10k": target_report(counts, "panel_cp10k_log1p"),
        }
        r = per_fold[name]
        print(f"    {name:12s} {len(r['sections'])} sections "
              f"spots={r['raw']['n_spots']:5d} genes={r['raw']['n_genes']:4d} "
              f"depth_med={r['raw']['panel_depth_median']:7.0f} "
              f"zero_depth={r['raw']['n_zero_depth_spots']:3d} "
              f"| raw max {r['raw']['target_max']:.2f} "
              f"-> cp10k max {r['panel_cp10k']['target_max']:.2f}")

    n_genes = next(iter(counts_by_fold.values())).shape[1]
    if panel is not None and len(panel) != n_genes:
        raise SystemExit(f"panel file has {len(panel)} genes but arrays have {n_genes}")
    if genes_ref is not None and panel is not None and genes_ref != panel:
        print("    WARNING: stored gene order differs from the panel file order")
    report["checks"]["counts_recovered"] = True
    report["n_genes"] = int(n_genes)
    report["folds"] = per_fold

    # ------------------------------------------------- 2. parity with score.py
    parity = {"mode": "reference_only"}
    if args.score_module:
        mod_name, _, fn_name = args.score_module.partition(":")
        fn = getattr(importlib.import_module(mod_name), fn_name)
        probe = next(iter(counts_by_fold.values()))[:512]
        mine = panel_cp10k_log1p(probe)
        theirs = np.asarray(fn(probe), dtype=np.float32)
        d = float(np.abs(mine - theirs).max())
        parity = {"mode": "checked", "callable": args.score_module, "max_abs_diff": d,
                  "pass": bool(d < 1e-6)}
        print(f"[2] parity vs {args.score_module}: max|diff| = {d:.3g} "
              f"-> {'PASS' if d < 1e-6 else 'FAIL'}")
        if d >= 1e-6:
            raise SystemExit("target definition does not match the scorer - stop here")
    else:
        print("[2] parity: no --score_module given. Confirm by hand that score.py's "
              "panel_cp10k renorm is  log1p(c / c.sum(axis=1) * 1e4)  with natural log, "
              "panel-scoped denominator, computed per spot after zero-fill.")
    report["checks"]["scorer_parity"] = parity

    # --------------------------------------------------------- 3/4. demo prior
    hold = args.holdout or sorted(counts_by_fold)[0]
    train_counts = np.concatenate(
        [c for k, c in sorted(counts_by_fold.items()) if k != hold], axis=0)
    y_train = build_target(train_counts, "panel_cp10k_log1p")
    prior = GaussianPrior.fit(y_train, sd_floor=args.sd_floor, genes=genes_ref or panel)
    s = prior.summary()
    print(f"[3] demo prior (train = all folds except {hold}): "
          f"{s['n_train_spots']} spots, mu in [{s['mu_min']:.3f}, {s['mu_max']:.3f}], "
          f"sd median {s['sd_median']:.3f}, {s['n_genes_at_sd_floor']} genes at the sd floor")
    if s["n_genes_at_sd_floor"] > 0.05 * n_genes:
        print("    WARNING: >5% of genes have no variance in training. Check zero-fill.")
    report["checks"]["demo_prior"] = s

    # ------------------------------------------------------ 5. depth oracle
    held = counts_by_fold[hold]
    prof = train_counts.sum(0)
    prof = prof / prof.sum()
    depth = held.sum(1, keepdims=True)
    oracle_counts = depth * prof[None, :]

    r_raw = per_gene_pcc(np.log1p(oracle_counts), np.log1p(held))
    orc_cp = panel_cp10k_log1p(oracle_counts)
    truth_cp = panel_cp10k_log1p(held)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r_cp = per_gene_pcc(orc_cp, truth_cp)
    sd_orc = float(np.median(orc_cp.std(0)))

    n_def = int(np.isfinite(r_cp).sum())
    cp_mean = float(np.nanmean(r_cp)) if n_def else None
    cp_txt = f"{cp_mean:+.3f}" if n_def else "undefined (constant predictor)"
    print(f"[4] depth-only oracle on fold {hold}: "
          f"raw log1p PCC mean {np.nanmean(r_raw):+.3f} | "
          f"panel_cp10k PCC mean {cp_txt} "
          f"(defined for {n_def}/{n_genes} genes), "
          f"median oracle SD on new footing {sd_orc:.2e}")
    print("    Expected: high on raw, ~0 or undefined on panel_cp10k - the oracle "
          "becomes a per-gene constant once depth is divided out. That collapse is "
          "the reason for this retrain; record it as the control.")
    report["checks"]["depth_oracle"] = {
        "fold": hold,
        "raw_pcc_mean": float(np.nanmean(r_raw)),
        "panel_cp10k_pcc_mean": cp_mean,
        "panel_cp10k_pcc_defined": n_def,
        "panel_cp10k_oracle_sd_median": sd_orc,
    }

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "preflight.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        prior.save(os.path.join(args.out, "demo_prior.npz"))
        print(f"\nwrote {os.path.join(args.out, 'preflight.json')}")

    print("\nPREFLIGHT OK")


if __name__ == "__main__":
    main()
