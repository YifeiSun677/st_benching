#!/usr/bin/env python
"""
compare_repro.py - did a HisToGene LOPO rerun reproduce the reference run,
and does it look like the COLOUR run or a GRAY arm?

Reads <run>/fold*_<P>/preds/<section>.npz (keys: pred, truth, spot_id, genes)
and <run>/fold*_<P>/run.json from both runs. Works on partial runs
(folds missing from the rerun are skipped), so it can be run after fold A.

Usage
  python compare_repro.py --ref /workspace/runs/histogene_lopo_833 \
                          --new /workspace/runs/histogene_lopo_833_rerun
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# per-fold per-gene PCC from the filed grayscale ablation (tag htg_gray_lopo_833)
PROFILES = {
    "COLOUR (colour-train / colour-test)": dict(A=0.0266, B=0.1208, C=0.0927, D=0.0987, E=-0.0070, F=0.0005, G=0.1011, H=0.1025),
    "GRAY arm2 (colour-train / gray-test)": dict(A=0.0198, B=0.0913, C=0.0505, D=0.0621, E=-0.0033, F=0.0017, G=0.0634, H=0.0860),
    "GRAY arm3 (gray-train / gray-test)": dict(A=0.0199, B=0.0629, C=0.0359, D=0.0478, E=-0.0036, F=0.0017, G=0.0428, H=0.0609),
}
EXACT_TOL = 1e-6      # max |delta pred| for a bitwise-level match
PCC_TOL = 0.002       # per-fold PCC tolerance for a numerical match
EXPECTED_DIFF_KEYS = ("time", "elapsed", "duration", "date", "start", "end", "wall",
                      "tag", "out", "run_dir", "host", "peak", "mem", "sec")


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def per_gene_pcc(P, T):
    P = P - P.mean(0)
    T = T - T.mean(0)
    den = np.sqrt((P ** 2).sum(0) * (T ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (P * T).sum(0) / den


def fold_dirs(run):
    return {os.path.basename(d).split("_")[-1]: d for d in sorted(glob.glob(os.path.join(run, "fold*_*")))}


def fold_pcc(fdir):
    files = sorted(glob.glob(os.path.join(fdir, "preds", "*.npz")))
    if not files:
        return None
    P = np.concatenate([np.load(f)["pred"] for f in files])
    T = np.concatenate([np.load(f)["truth"] for f in files])
    return float(np.nanmean(per_gene_pcc(P.astype(np.float64), T.astype(np.float64))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    a = ap.parse_args()

    ref, new = fold_dirs(a.ref), fold_dirs(a.new)
    if not new:
        sys.exit(f"No fold dirs in {a.new}")
    folds = [p for p in sorted(ref) if p in new]
    print(f"Folds in both runs: {' '.join(folds)}   (missing from rerun: {' '.join(sorted(set(ref) - set(new))) or 'none'})")

    # ---------- 1. config ----------
    print("\n=== 1. run.json config differences (fold " + folds[0] + ") ===")
    try:
        cr = flatten(json.load(open(os.path.join(ref[folds[0]], "run.json"))))
        cn = flatten(json.load(open(os.path.join(new[folds[0]], "run.json"))))
        diffs = [k for k in sorted(set(cr) | set(cn)) if cr.get(k) != cn.get(k)]
        unexpected = []
        for k in diffs:
            tag = "expected " if any(s in k.lower() for s in EXPECTED_DIFF_KEYS) else "CHECK    "
            if tag.startswith("CHECK"):
                unexpected.append(k)
            print(f"  [{tag}] {k}: ref={str(cr.get(k))[:70]}  new={str(cn.get(k))[:70]}")
        if not diffs:
            print("  identical")
        gray_keys = {k: v for k, v in cn.items() if "gray" in k.lower() or "grey" in k.lower()}
        print(f"  gray-related keys in rerun: {gray_keys or 'none'}")
    except Exception as e:
        unexpected = ["run.json unreadable"]
        print(f"  could not compare run.json: {e}")

    # ---------- 2. predictions ----------
    print("\n=== 2. prediction-level comparison ===")
    worst_pred, ids_ok, truth_ok, sections_missing = 0.0, True, True, 0
    for p in folds:
        for f in sorted(glob.glob(os.path.join(ref[p], "preds", "*.npz"))):
            g = os.path.join(new[p], "preds", os.path.basename(f))
            if not os.path.exists(g):
                print(f"  MISSING {g}")
                sections_missing += 1
                continue
            A, B = np.load(f), np.load(g)
            same_ids = A["spot_id"].shape == B["spot_id"].shape and bool((A["spot_id"] == B["spot_id"]).all())
            same_genes = bool((A["genes"] == B["genes"]).all())
            dt = float(np.abs(A["truth"] - B["truth"]).max()) if A["truth"].shape == B["truth"].shape else np.inf
            dp = float(np.abs(A["pred"] - B["pred"]).max()) if A["pred"].shape == B["pred"].shape else np.inf
            ids_ok &= same_ids and same_genes
            truth_ok &= dt == 0.0
            worst_pred = max(worst_pred, dp)
            print(f"  {p} {os.path.basename(f):10s} ids={same_ids} genes={same_genes} "
                  f"max|dtruth|={dt:.1e} max|dpred|={dp:.2e}")

    # ---------- 3. per-fold PCC ----------
    print("\n=== 3. per-fold per-gene PCC ===")
    print(f"  {'fold':4s} {'ref':>8s} {'rerun':>8s} {'delta':>8s} {'colour':>8s} {'gray3':>8s}")
    pr, pn, max_dpcc = {}, {}, 0.0
    for p in folds:
        pr[p], pn[p] = fold_pcc(ref[p]), fold_pcc(new[p])
        d = pn[p] - pr[p]
        max_dpcc = max(max_dpcc, abs(d))
        c = list(PROFILES.values())
        print(f"  {p:4s} {pr[p]:8.4f} {pn[p]:8.4f} {d:+8.4f} {c[0].get(p, np.nan):8.4f} {c[2].get(p, np.nan):8.4f}")
    print(f"  mean {np.mean(list(pr.values())):8.4f} {np.mean(list(pn.values())):8.4f}")

    # ---------- 4. which profile ----------
    print("\n=== 4. nearest known profile ===")
    dist = {}
    for name, prof in PROFILES.items():
        for label, vals in (("ref", pr), ("rerun", pn)):
            dist[(label, name)] = float(np.sqrt(np.mean([(vals[p] - prof[p]) ** 2 for p in folds])))
    for label in ("ref", "rerun"):
        best = min((k for k in dist if k[0] == label), key=dist.get)
        print(f"  {label:5s} -> {best[1]}  (rms {dist[best]:.4f})")
        for k in dist:
            if k[0] == label:
                print(f"          {k[1]:40s} rms {dist[k]:.4f}")
    ref_is_colour = min((k for k in dist if k[0] == "ref"), key=dist.get)[1].startswith("COLOUR")
    new_is_colour = min((k for k in dist if k[0] == "rerun"), key=dist.get)[1].startswith("COLOUR")

    # ---------- verdict ----------
    print("\n=== VERDICT ===")
    if not ref_is_colour:
        print("  WARNING: the REFERENCE run does not match the filed colour profile - "
              "check the --ref path, or the PCC footing differs from the filed table.")
    if sections_missing or not ids_ok or not truth_ok:
        print(f"  FAIL - structural mismatch (missing sections {sections_missing}, ids/genes ok {ids_ok}, truth identical {truth_ok})")
    elif worst_pred <= EXACT_TOL:
        print(f"  REPRODUCED EXACTLY (max |dpred| {worst_pred:.1e})")
    elif max_dpcc <= PCC_TOL:
        print(f"  REPRODUCED NUMERICALLY (max |dpred| {worst_pred:.1e}, max |dPCC| {max_dpcc:.4f}) - "
              "not bitwise; usual cause is a different GPU type / nondeterministic kernels")
    else:
        print(f"  NOT REPRODUCED (max |dPCC| {max_dpcc:.4f} > {PCC_TOL})")
    print(f"  Input looks: {'COLOUR' if new_is_colour else 'GRAY'}")
    if unexpected:
        print(f"  Config keys to check by eye: {', '.join(unexpected)}")
    if len(folds) < 8:
        print(f"  (partial check: {len(folds)}/8 folds)")


if __name__ == "__main__":
    main()
