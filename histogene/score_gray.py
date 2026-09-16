"""Score the HisToGene colour ablation (three arms), fold by fold.

    python -m histogene.score_gray --tag htg_gray_lopo_833
    python -m histogene.score_gray --tag htg_gray_lopo_833 --ref_tag histogene_lopo_833

Headline = per-gene PCC computed INSIDE each LOPO fold (the held-out patient's
sections concatenated), averaged over genes - the same footing as the filed
HisToGene number (+0.067). Never pooled across folds.

Per fold and arm it also reports
  pcc_section_mean   per-gene PCC per section, averaged over sections then genes
  frac_pos           fraction of genes with PCC > 0 (fold unit)
  sse_ratio_median   per gene: SSE(model) / SSE(that section's own gene mean),
                     summed over the fold's sections; median over genes
  frac_beat_baseline fraction of genes with that ratio < 1
  sd_ratio_median    per gene: within-section pred SD / true SD; median over genes

Paired contrasts over the 8 folds
  arm3 - arm1   does colour carry information?          (retrained)
  arm2 - arm1   the existing test-time perturbation drop (colour model, gray input)
  arm3 - arm2   how much of that drop is train/test MISMATCH
  recovery      (arm3 - arm2) / (arm1 - arm2)

Refuses to score if: an arm is missing a fold, gray checksums disagree or are
not the canonical one, ground truth / spot ids differ between arms, or arm2's
predictions equal arm1's.

Writes results/<tag>/{per_fold.csv, per_section.csv, per_gene.csv,
paired.csv, summary.json}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .gray_bridge import EXPECTED_CHECKSUM

ARMS = ["arm1_colour", "arm2_graytest", "arm3_graytrain"]
SHORT = {"arm1_colour": "arm1", "arm2_graytest": "arm2", "arm3_graytrain": "arm3"}
EXPECT_MODE = {"arm1_colour": "none", "arm2_graytest": "query", "arm3_graytrain": "all"}
MARKERS = ["ERBB2", "GRB7", "ESR1", "PGR", "FASN", "GNAS", "MKI67"]
METRICS = ["pcc_fold_mean", "pcc_fold_median", "pcc_section_mean", "frac_pos",
           "sse_ratio_median", "frac_beat_baseline", "sd_ratio_median"]


# ---------------------------------------------------------------- maths -----
def pcc_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    num = (A * B).sum(0)
    den = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0))
    out = np.full(A.shape[1], np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def fold_metrics(secs: dict) -> tuple[dict, np.ndarray, list]:
    """secs: {section: (pred, truth)} for one fold of one arm."""
    P = np.concatenate([v[0] for v in secs.values()]).astype(np.float64)
    T = np.concatenate([v[1] for v in secs.values()]).astype(np.float64)
    g_fold = pcc_cols(P, T)

    sec_rows, sec_pcc = [], []
    G = P.shape[1]
    sse_m, sse_b = np.zeros(G), np.zeros(G)
    var_p, var_t = np.zeros(G), np.zeros(G)
    for s, (p, t) in secs.items():
        p = p.astype(np.float64)
        t = t.astype(np.float64)
        r = pcc_cols(p, t)
        sec_pcc.append(r)
        mu = t.mean(0, keepdims=True)
        sm, sb = ((p - t) ** 2).sum(0), ((t - mu) ** 2).sum(0)
        sse_m += sm
        sse_b += sb
        vp = ((p - p.mean(0, keepdims=True)) ** 2).sum(0)
        var_p += vp
        var_t += sb
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_s = np.where(sb > 0, sm / sb, np.nan)
        sec_rows.append({"section": s, "n_spots": len(t),
                         "pcc_mean": float(np.nanmean(r)),
                         "pcc_median": float(np.nanmedian(r)),
                         "frac_pos": float(np.nanmean(r > 0)),
                         "sse_ratio_median": float(np.nanmedian(ratio_s))})
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sse_b > 0, sse_m / sse_b, np.nan)
        sd_ratio = np.where(var_t > 0, np.sqrt(var_p / var_t), np.nan)
    valid = ~np.isnan(ratio)
    m = {
        "n_sections": len(secs), "n_spots": len(T),
        "pcc_fold_mean": float(np.nanmean(g_fold)),
        "pcc_fold_median": float(np.nanmedian(g_fold)),
        "pcc_section_mean": float(np.nanmean(np.nanmean(np.vstack(sec_pcc), axis=0))),
        "frac_pos": float(np.nanmean(g_fold > 0)),
        "sse_ratio_median": float(np.nanmedian(ratio)),
        "frac_beat_baseline": float((ratio[valid] < 1).mean()) if valid.any() else np.nan,
        "sd_ratio_median": float(np.nanmedian(sd_ratio)),
    }
    return m, g_fold, sec_rows


# ------------------------------------------------------------------ io ------
def load_arm(arm_dir: Path) -> dict:
    """{fold_dir: {"meta": run.json, "secs": {sec: dict(pred,truth,spot_id)}, "genes"}}"""
    out = {}
    for fd in sorted(p for p in arm_dir.iterdir() if p.is_dir()):
        rj = fd / "run.json"
        files = sorted((fd / "preds").glob("*.npz"))
        if not rj.exists() or not files:
            continue
        secs, genes = {}, None
        for f in files:
            z = np.load(f, allow_pickle=True)
            secs[f.stem] = {"pred": z["pred"], "truth": z["truth"],
                            "spot_id": [str(x) for x in z["spot_id"]]}
            genes = [str(g) for g in z["genes"]]
        out[fd.name] = {"meta": json.loads(rj.read_text()), "secs": secs, "genes": genes}
    return out


def load_ref(ref_root: Path) -> dict:
    """Original colour run (histogene/train.py layout): fold dirs with preds/."""
    out = {}
    for fd in sorted(p for p in ref_root.iterdir() if p.is_dir() and (p / "preds").is_dir()):
        secs = {}
        for f in sorted((fd / "preds").glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            secs[f.stem] = (z["pred"], z["truth"])
        if secs:
            out[fd.name] = secs
    return out


# --------------------------------------------------------------- stats ------
def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    res = {"n": n, "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if n > 1 else np.nan,
           "n_up": int((d > 0).sum()), "n_down": int((d < 0).sum()),
           "ci95_lo": np.nan, "ci95_hi": np.nan, "p_ttest": np.nan, "p_wilcoxon": np.nan}
    if n > 1:
        try:
            from scipy import stats
            se = d.std(ddof=1) / np.sqrt(n)
            q = stats.t.ppf(0.975, n - 1)
            res["ci95_lo"], res["ci95_hi"] = float(d.mean() - q * se), float(d.mean() + q * se)
            res["p_ttest"] = float(stats.ttest_1samp(d, 0.0).pvalue)
            if np.any(d != 0):
                res["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue)
        except ImportError:
            pass
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref_tag", default=None,
                    help="original colour run under the runs dir, e.g. histogene_lopo_833")
    ap.add_argument("--gene_sets", default=str(C.GENE_SETS_DIR))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = C.OUT_DIR / args.tag
    runs = {}
    for arm in ARMS:
        if not (root / arm).is_dir():
            raise SystemExit(f"missing arm directory {root / arm}")
        runs[arm] = load_arm(root / arm)
    folds = sorted(runs["arm1_colour"])
    if not folds:
        raise SystemExit("no finished folds")

    # ---------------------------------------------------- consistency -----
    print("=== consistency checks ===")
    problems = []
    for arm in ARMS:
        if sorted(runs[arm]) != folds:
            problems.append(f"{arm} folds {sorted(runs[arm])} != arm1 {folds}")
    if problems:
        raise SystemExit("\n".join(problems))
    genes = runs["arm1_colour"][folds[0]]["genes"]
    cks = {}
    for arm in ARMS:
        for f in folds:
            m = runs[arm][f]["meta"]
            cks.setdefault(arm, set()).add(m.get("gray_checksum"))
            if m.get("gray_mode") != EXPECT_MODE[arm]:
                problems.append(f"{arm}/{f}: gray_mode {m.get('gray_mode')}")
            if runs[arm][f]["genes"] != genes:
                problems.append(f"{arm}/{f}: gene order differs")
            chk = m.get("test_input_check", {})
            want_gray = arm != "arm1_colour"
            if want_gray and chk.get("channel_spread_max", 1) != 0:
                problems.append(f"{arm}/{f}: test input not gray")
            if not want_gray and chk.get("channel_spread_max", 0) == 0:
                problems.append(f"{arm}/{f}: colour test input has no colour")
    all_ck = set().union(*cks.values())
    print(f"  gray checksums seen: {sorted(map(str, all_ck))}  (expect {EXPECTED_CHECKSUM})")
    if all_ck != {EXPECTED_CHECKSUM}:
        problems.append(f"checksums {all_ck} != {{{EXPECTED_CHECKSUM}}}")
    for f in folds:
        s1 = runs["arm1_colour"][f]["secs"]
        for arm in ARMS[1:]:
            sa = runs[arm][f]["secs"]
            if sorted(sa) != sorted(s1):
                problems.append(f"{arm}/{f}: sections differ")
                continue
            for s in s1:
                if sa[s]["spot_id"] != s1[s]["spot_id"]:
                    problems.append(f"{arm}/{f}/{s}: spot order differs")
                if not np.array_equal(sa[s]["truth"], s1[s]["truth"]):
                    problems.append(f"{arm}/{f}/{s}: truth differs")
        for s in s1:
            d12 = float(np.abs(runs["arm2_graytest"][f]["secs"][s]["pred"]
                               - s1[s]["pred"]).max())
            if d12 == 0:
                problems.append(f"{f}/{s}: arm2 pred == arm1 pred (gray had no effect)")
    if problems:
        raise SystemExit("REFUSING TO SCORE:\n  " + "\n  ".join(problems))
    print(f"  OK: {len(folds)} folds x 3 arms, same sections/spots/truth, "
          f"one checksum, arm2 != arm1")

    # ------------------------------------------------------- metrics ------
    fold_rows, sec_rows, gene_pcc = [], [], {a: [] for a in ARMS}
    for arm in ARMS:
        for f in folds:
            secs = {s: (d["pred"], d["truth"]) for s, d in runs[arm][f]["secs"].items()}
            m, g, srows = fold_metrics(secs)
            meta = runs[arm][f]["meta"]
            fold_rows.append({"arm": SHORT[arm], "fold": f,
                              "patient": meta.get("held_out", f.split("_")[-1]), **m,
                              "final_train_loss": meta.get("final_train_loss"),
                              "sec_per_epoch": meta.get("sec_per_epoch")})
            for r in srows:
                sec_rows.append({"arm": SHORT[arm], "fold": f, **r})
            gene_pcc[arm].append(g)
    fold_df = pd.DataFrame(fold_rows)
    sec_df = pd.DataFrame(sec_rows)

    wide = fold_df.pivot(index="patient", columns="arm", values="pcc_fold_mean")
    print("\n=== per-fold per-gene PCC (mean over 833 genes) ===")
    print(wide.assign(d31=wide["arm3"] - wide["arm1"],
                      d21=wide["arm2"] - wide["arm1"],
                      d32=wide["arm3"] - wide["arm2"])
          .to_string(float_format="%+.4f"))

    # --------------------------------------------------------- paired -----
    prs, summary_pairs = [], {}
    for metric in METRICS:
        w = fold_df.pivot(index="fold", columns="arm", values=metric).loc[folds]
        for name, a, b in [("arm3-arm1", "arm3", "arm1"),
                           ("arm2-arm1", "arm2", "arm1"),
                           ("arm3-arm2", "arm3", "arm2")]:
            r = paired(w[a].values, w[b].values)
            prs.append({"metric": metric, "contrast": name, **r})
            if metric == "pcc_fold_mean":
                summary_pairs[name] = r
    pair_df = pd.DataFrame(prs)
    arm_means = fold_df.groupby("arm")[METRICS].mean()
    drop = arm_means.loc["arm1", "pcc_fold_mean"] - arm_means.loc["arm2", "pcc_fold_mean"]
    rec = arm_means.loc["arm3", "pcc_fold_mean"] - arm_means.loc["arm2", "pcc_fold_mean"]
    recovery = float(rec / drop) if drop != 0 else np.nan

    print("\n=== arm means over folds ===")
    print(arm_means.to_string(float_format="%.4f"))
    print("\n=== paired contrasts, pcc_fold_mean ===")
    print(pair_df[pair_df.metric == "pcc_fold_mean"]
          .drop(columns="metric").to_string(index=False, float_format="%+.4f"))
    print(f"\nrecovery (arm3-arm2)/(arm1-arm2) = {recovery:.2f}")

    # ---------------------------------------------------------- genes -----
    gm = {SHORT[a]: np.nanmean(np.vstack(gene_pcc[a]), axis=0) for a in ARMS}
    gene_df = pd.DataFrame({"gene": genes, **{f"pcc_{k}": v for k, v in gm.items()}})
    gene_df["d31"] = gene_df["pcc_arm3"] - gene_df["pcc_arm1"]
    gene_df["d21"] = gene_df["pcc_arm2"] - gene_df["pcc_arm1"]

    summary = {
        "model": "HisToGene", "tag": args.tag, "n_folds": len(folds), "n_genes": len(genes),
        "gray_checksum": EXPECTED_CHECKSUM,
        "epochs": runs["arm1_colour"][folds[0]]["meta"].get("epochs"),
        "lr": runs["arm1_colour"][folds[0]]["meta"].get("lr"),
        "arm_means": arm_means.to_dict(orient="index"),
        "fold_range_pcc": {k: [float(fold_df[fold_df.arm == k].pcc_fold_mean.min()),
                               float(fold_df[fold_df.arm == k].pcc_fold_mean.max())]
                           for k in ("arm1", "arm2", "arm3")},
        "paired_pcc_fold_mean": summary_pairs,
        "recovery_fraction": recovery,
        "markers": {m: {k: float(gm[k][genes.index(m)]) for k in gm}
                    for m in MARKERS if m in genes},
    }

    gs_dir = Path(args.gene_sets).expanduser()
    if gs_dir.is_dir():
        strata = {}
        for nm in ("all", "hvg", "svg", "marker"):
            f = gs_dir / f"gene_set_{nm}.txt"
            if f.exists():
                idx = [genes.index(g) for g in f.read_text().split() if g in genes]
                if idx:
                    strata[nm] = {"n": len(idx),
                                  **{k: float(np.nanmean(v[idx])) for k, v in gm.items()}}
        summary["gene_sets"] = strata
    else:
        print(f"[note] {gs_dir} not found - gene-set strata skipped")

    if args.ref_tag:
        ref_root = C.OUT_DIR / args.ref_tag
        if ref_root.is_dir():
            ref = load_ref(ref_root)
            rows = []
            for f in folds:
                if f in ref:
                    m, _, _ = fold_metrics(ref[f])
                    a1 = float(fold_df[(fold_df.arm == "arm1") & (fold_df.fold == f)]
                               .pcc_fold_mean.iloc[0])
                    rows.append({"fold": f, "original": m["pcc_fold_mean"], "arm1": a1,
                                 "diff": a1 - m["pcc_fold_mean"]})
            if rows:
                rdf = pd.DataFrame(rows)
                print(f"\n=== reproduction: arm1 vs original run ({args.ref_tag}) ===")
                print(rdf.to_string(index=False, float_format="%+.4f"))
                summary["reproduction_vs_ref"] = {
                    "ref_tag": args.ref_tag,
                    "ref_mean": float(rdf.original.mean()),
                    "arm1_mean": float(rdf.arm1.mean()),
                    "max_abs_fold_diff": float(rdf["diff"].abs().max())}
        else:
            print(f"[note] ref run {ref_root} not found - reproduction check skipped")

    out = Path(args.out) if args.out else C.RESULTS_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out / "per_fold.csv", index=False)
    sec_df.to_csv(out / "per_section.csv", index=False)
    gene_df.to_csv(out / "per_gene.csv", index=False)
    pair_df.to_csv(out / "paired.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
