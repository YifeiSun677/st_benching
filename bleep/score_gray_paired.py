#!/usr/bin/env python
"""Paired three-arm scoring for the BLEEP patient-B colour ablation.

Arms
  1  colour train  -> colour test      (finished: per-gene PCC +0.366)
  2  colour train  -> grayscale test   (perturbation control, inference only)
  3  grayscale train -> grayscale test (the new arm)

The unit of analysis is the held-out SECTION. Patient B has 6 of them, so
every comparison is a paired difference over n = 6. All per-gene PCCs are
computed within a section and then averaged across sections -- never pooled
across sections first, which on her2st inverts the sign of the real signal.

Alignment and PCC are reported separately and never combined into a score.
Alignment says whether the joint embedding still puts a patch near its own
expression profile; PCC says whether the retrieved expression is any good.
BLEEP is the one model in this benchmark where those can come apart, and that
separation is the entire reason it is worth running this ablation on BLEEP.

Usage
  python -m bleep.score_gray_paired \
      --arm1 /workspace/runs/bleep_patientB_833 \
      --arm2 /workspace/runs/bleep_patientB_833_gray_test \
      --arm3 /workspace/runs/bleep_patientB_833_gray_train \
      --out  /workspace/results/bleep_gray_patientB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover
    sps = None

ARM_LABELS = {
    "arm1": "colour_train_colour_test",
    "arm2": "colour_train_gray_test",
    "arm3": "gray_train_gray_test",
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def find_fold_dirs(run_dir: Path) -> list[Path]:
    """Any subdirectory holding a preds.npz, sorted by name."""
    folds = sorted({p.parent for p in run_dir.rglob("preds.npz")})
    if not folds:
        raise FileNotFoundError(f"no preds.npz found anywhere under {run_dir}")
    return folds


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def extract_alignment(diag: dict) -> float | None:
    """Pull the retrieval alignment number out of diagnostics.json.

    The key name has moved around between runs, so match on substring rather
    than hard-coding it. Chance is ~50 for this metric.
    """
    for key, value in diag.items():
        if "align" in key.lower() and isinstance(value, (int, float)):
            return float(value)
    for key, value in diag.items():
        if isinstance(value, dict):
            inner = extract_alignment(value)
            if inner is not None:
                return inner
    return None


def load_arm(run_dir: Path, arm: str) -> dict:
    """Return {section: {pred, truth, genes, alignment, fold_dir}} for one arm."""
    out: dict[str, dict] = {}
    checksums: set[str] = set()
    epochs: set = set()

    for fold_dir in find_fold_dirs(run_dir):
        npz = np.load(fold_dir / "preds.npz", allow_pickle=True)
        pred = np.asarray(npz["pred"], dtype=np.float64)
        truth = np.asarray(npz["truth"], dtype=np.float64)
        genes = np.asarray(npz["genes"]).astype(str)
        qkeys = np.asarray(npz["query_keys"]).astype(str)

        cfg = read_json(fold_dir / "run.json")
        diag = read_json(fold_dir / "diagnostics.json")
        alignment = extract_alignment(diag)
        if "gray_checksum" in cfg:
            checksums.add(str(cfg["gray_checksum"]))
        for k in ("epochs", "n_epochs", "epoch"):
            if k in cfg:
                epochs.add(cfg[k])

        # query_keys look like "B1:10x13" -- the prefix splits rows with no join
        sections = np.array([k.split(":")[0] for k in qkeys])
        for sec_raw in np.unique(sections):
            sec = str(sec_raw)
            mask = sections == sec_raw
            if sec in out:
                raise ValueError(
                    f"{arm}: section {sec} appears in more than one fold of {run_dir}")
            out[sec] = {
                "pred": pred[mask],
                "truth": truth[mask],
                "genes": genes,
                "alignment": alignment,
                "fold_dir": str(fold_dir),
                "n_spots": int(mask.sum()),
            }

    return {
        "sections": out,
        "checksums": checksums,
        "epochs": epochs,
        "run_dir": str(run_dir),
        "label": ARM_LABELS.get(arm, arm),
    }


# --------------------------------------------------------------------------
# metrics, all computed within one section
# --------------------------------------------------------------------------
def per_gene_pcc(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-gene Pearson r across the spots of one section. NaN where undefined."""
    p = pred - pred.mean(axis=0, keepdims=True)
    t = truth - truth.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    den = np.sqrt((p**2).sum(axis=0) * (t**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return r


def section_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    r = per_gene_pcc(pred, truth)
    valid = np.isfinite(r)

    # baseline = that gene's own mean in that section, i.e. the L2-optimal
    # constant predictor. ratio < 1 means the model beat doing nothing.
    resid = ((pred - truth) ** 2).sum(axis=0)
    null = ((truth - truth.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sse_ratio = np.where(null > 0, resid / null, np.nan)

    sd_p = pred.std(axis=0)
    sd_t = truth.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd_ratio = np.where(sd_t > 0, sd_p / sd_t, np.nan)

    return {
        "n_spots": int(pred.shape[0]),
        "n_genes_valid": int(valid.sum()),
        "pcc_mean": float(np.nanmean(r)),
        "pcc_median": float(np.nanmedian(r)),
        "frac_pcc_positive": float(np.nanmean(r > 0)),
        "sse_ratio_median": float(np.nanmedian(sse_ratio)),
        "frac_beat_baseline": float(np.nanmean(sse_ratio < 1)),
        "sd_ratio_median": float(np.nanmedian(sd_ratio)),
        "_per_gene_pcc": r,
    }


# --------------------------------------------------------------------------
# paired statistics over sections
# --------------------------------------------------------------------------
def paired_report(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> dict:
    """b - a, paired over sections."""
    d = b - a
    n = len(d)
    res = {
        "comparison": f"{name_b} - {name_a}",
        "n_sections": n,
        "mean_delta": float(d.mean()),
        "sd_delta": float(d.std(ddof=1)) if n > 1 else float("nan"),
        "median_delta": float(np.median(d)),
        "min_delta": float(d.min()),
        "max_delta": float(d.max()),
        "n_sections_delta_positive": int((d > 0).sum()),
        "per_section_delta": [float(x) for x in d],
    }
    if sps is not None and n > 1:
        t_stat, t_p = sps.ttest_rel(b, a)
        res["paired_t_p"] = float(t_p)
        res["paired_t_stat"] = float(t_stat)
        if np.any(d != 0):
            try:
                w_stat, w_p = sps.wilcoxon(b, a, zero_method="wilcox", mode="exact")
            except TypeError:  # newer scipy renamed the argument
                w_stat, w_p = sps.wilcoxon(b, a, zero_method="wilcox",
                                           method="exact")
            res["wilcoxon_p"] = float(w_p)
            res["wilcoxon_stat"] = float(w_stat)
        res["wilcoxon_min_attainable_p"] = float(2.0 / (2**n))
    return res


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm1", required=True, help="colour train / colour test run dir")
    ap.add_argument("--arm2", required=True, help="colour train / gray test run dir")
    ap.add_argument("--arm3", required=True, help="gray train / gray test run dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-checksum-mismatch", action="store_true",
                    help="escape hatch; do not use for a reportable result")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    arms = {
        "arm1": load_arm(Path(args.arm1), "arm1"),
        "arm2": load_arm(Path(args.arm2), "arm2"),
        "arm3": load_arm(Path(args.arm3), "arm3"),
    }

    # ---- guard 1: the two grayscale arms must share one transform ---------
    cs2 = arms["arm2"]["checksums"]
    cs3 = arms["arm3"]["checksums"]
    print("grayscale checksums   arm2:", cs2 or "{MISSING}", " arm3:", cs3 or "{MISSING}")
    if not cs2 or not cs3:
        msg = ("run.json is missing gray_checksum in at least one grayscale arm; "
               "merge gray_provenance() into run.json and rerun")
        if not args.allow_checksum_mismatch:
            print("FATAL:", msg)
            return 2
        print("WARNING:", msg)
    elif cs2 != cs3:
        msg = f"arm 2 and arm 3 used DIFFERENT grayscale transforms: {cs2} vs {cs3}"
        if not args.allow_checksum_mismatch:
            print("FATAL:", msg, "-- the three numbers are not comparable")
            return 2
        print("WARNING:", msg)

    # ---- guard 2: same sections, same gene order -------------------------
    secs = sorted(set(arms["arm1"]["sections"]))
    for key, arm in arms.items():
        got = sorted(set(arm["sections"]))
        if got != secs:
            print(f"FATAL: {key} has sections {got}, arm1 has {secs}")
            return 2
    ref_genes = arms["arm1"]["sections"][secs[0]]["genes"]
    for key, arm in arms.items():
        for sec in secs:
            g = arm["sections"][sec]["genes"]
            if len(g) != len(ref_genes) or not np.array_equal(g, ref_genes):
                print(f"FATAL: gene order differs in {key}/{sec}")
                return 2
    print(f"sections: {secs}    genes: {len(ref_genes)}")
    for key, arm in arms.items():
        print(f"  {key:5s} {arm['label']:32s} epochs={sorted(arm['epochs']) or '?'}")

    # ---- per-section metrics ---------------------------------------------
    rows = []
    per_gene = {k: [] for k in arms}
    for key, arm in arms.items():
        for sec in secs:
            d = arm["sections"][sec]
            m = section_metrics(d["pred"], d["truth"])
            per_gene[key].append(m.pop("_per_gene_pcc"))
            rows.append({
                "arm": key, "label": arm["label"], "section": sec,
                "alignment": d["alignment"], **m,
            })

    import csv
    field_order = ["arm", "label", "section", "n_spots", "n_genes_valid",
                   "alignment", "pcc_mean", "pcc_median", "frac_pcc_positive",
                   "sse_ratio_median", "frac_beat_baseline", "sd_ratio_median"]
    with (out / "per_section_by_arm.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=field_order)
        w.writeheader()
        w.writerows(rows)

    # ---- printed summary --------------------------------------------------
    def col(key: str, metric: str) -> np.ndarray:
        return np.array([r[metric] for r in rows if r["arm"] == key
                         and r[metric] is not None], dtype=float)

    print("\n" + "=" * 78)
    print("PER-SECTION per-gene PCC (mean over 833 genes, computed within section)")
    print("=" * 78)
    header = f"{'section':>8} " + "".join(f"{k:>14}" for k in arms)
    print(header)
    for i, sec in enumerate(secs):
        line = f"{sec:>8} "
        for key in arms:
            line += f"{per_gene_mean(per_gene[key][i]):>14.4f}"
        print(line)
    line = f"{'MEAN':>8} "
    for key in arms:
        line += f"{np.mean([per_gene_mean(x) for x in per_gene[key]]):>14.4f}"
    print(line)

    print("\n" + "=" * 78)
    print("ALIGNMENT, reported separately (chance ~50)")
    print("=" * 78)
    print(header)
    for i, sec in enumerate(secs):
        line = f"{sec:>8} "
        for key in arms:
            a = rows[[j for j, r in enumerate(rows)
                      if r["arm"] == key and r["section"] == sec][0]]["alignment"]
            line += f"{'   n/a' if a is None else f'{a:14.2f}'}"
        print(line)

    print("\n" + "=" * 78)
    print("OTHER METRICS, mean over the 6 sections")
    print("=" * 78)
    for metric in ("pcc_median", "frac_pcc_positive", "sse_ratio_median",
                   "frac_beat_baseline", "sd_ratio_median"):
        line = f"{metric:>22} "
        for key in arms:
            line += f"{col(key, metric).mean():>14.4f}"
        print(line)

    # ---- paired comparisons ----------------------------------------------
    headline: dict = {"sections": secs, "n_genes": int(len(ref_genes))}
    comparisons = []
    for metric in ("pcc_mean", "sse_ratio_median", "frac_beat_baseline",
                   "sd_ratio_median", "alignment"):
        v = {k: col(k, metric) for k in arms}
        if any(len(v[k]) != len(secs) for k in arms):
            continue
        for a_key, b_key in (("arm1", "arm3"), ("arm1", "arm2"), ("arm2", "arm3")):
            rep = paired_report(v[a_key], v[b_key],
                                ARM_LABELS[a_key], ARM_LABELS[b_key])
            rep["metric"] = metric
            comparisons.append(rep)

    print("\n" + "=" * 78)
    print("PAIRED DIFFERENCES over the 6 sections")
    print("(n=6: the smallest two-sided Wilcoxon p attainable is 0.031, so read")
    print(" the sign consistency and the effect size, not the p value alone)")
    print("=" * 78)
    for rep in comparisons:
        if rep["metric"] != "pcc_mean":
            continue
        print(f"\n{rep['comparison']}   metric = per-gene PCC")
        print(f"  mean delta      {rep['mean_delta']:+.4f}  "
              f"(sd {rep['sd_delta']:.4f}, range {rep['min_delta']:+.4f} "
              f"to {rep['max_delta']:+.4f})")
        print(f"  sections up     {rep['n_sections_delta_positive']}/{rep['n_sections']}")
        print("  per section     " +
              "  ".join(f"{s}:{d:+.3f}" for s, d in zip(secs, rep["per_section_delta"])))
        if "wilcoxon_p" in rep:
            print(f"  paired t p      {rep['paired_t_p']:.4f}"
                  f"   wilcoxon p {rep['wilcoxon_p']:.4f}")

    with (out / "paired_deltas.json").open("w") as fh:
        json.dump(comparisons, fh, indent=2)

    # ---- per-gene paired deltas ------------------------------------------
    gene_mean = {k: np.nanmean(np.vstack(per_gene[k]), axis=0) for k in arms}
    d31 = gene_mean["arm3"] - gene_mean["arm1"]
    d21 = gene_mean["arm2"] - gene_mean["arm1"]
    with (out / "per_gene_pcc_by_arm.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gene", "arm1_colour_colour", "arm2_colour_gray",
                    "arm3_gray_gray", "delta_arm3_arm1", "delta_arm2_arm1"])
        for i, g in enumerate(ref_genes):
            w.writerow([g, f"{gene_mean['arm1'][i]:.6f}", f"{gene_mean['arm2'][i]:.6f}",
                        f"{gene_mean['arm3'][i]:.6f}", f"{d31[i]:.6f}", f"{d21[i]:.6f}"])

    order = np.argsort(np.nan_to_num(d31))
    print("\n" + "=" * 78)
    print("PER-GENE delta (gray_gray - colour_colour), 15 most hurt / 15 most helped")
    print("=" * 78)
    for i in order[:15]:
        print(f"  {ref_genes[i]:<14} {gene_mean['arm1'][i]:+.3f} -> "
              f"{gene_mean['arm3'][i]:+.3f}   {d31[i]:+.3f}")
    print("  ...")
    for i in order[-15:][::-1]:
        print(f"  {ref_genes[i]:<14} {gene_mean['arm1'][i]:+.3f} -> "
              f"{gene_mean['arm3'][i]:+.3f}   {d31[i]:+.3f}")

    headline["per_gene_pcc_by_arm"] = {
        k: float(np.mean([per_gene_mean(x) for x in per_gene[k]])) for k in arms}
    headline["alignment_by_arm"] = {}
    for k in arms:
        a = col(k, "alignment")
        headline["alignment_by_arm"][k] = float(a.mean()) if len(a) else None
    headline["paired"] = {r["metric"] + " | " + r["comparison"]: {
        "mean_delta": r["mean_delta"],
        "sections_up": f"{r['n_sections_delta_positive']}/{r['n_sections']}",
        "wilcoxon_p": r.get("wilcoxon_p"),
    } for r in comparisons}
    with (out / "headline.json").open("w") as fh:
        json.dump(headline, fh, indent=2)

    print(f"\nwrote: {out}/per_section_by_arm.csv")
    print(f"       {out}/paired_deltas.json")
    print(f"       {out}/per_gene_pcc_by_arm.csv")
    print(f"       {out}/headline.json")
    return 0


def per_gene_mean(r: np.ndarray) -> float:
    return float(np.nanmean(r))


if __name__ == "__main__":
    sys.exit(main())
