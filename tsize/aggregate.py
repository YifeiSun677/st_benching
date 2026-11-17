"""
tsize/aggregate.py -- collect the training-size runs into one tidy table + summaries.

    python -m tsize.aggregate                                        # pod defaults
    python -m tsize.aggregate --ridge-dir ../results_4_models/ridge_tsize \\
                              --triplex-dir ../results_4_models/triplex \\
                              --out ../results_4_models/tsize_summary     # on the Mac

Inputs
  ridge    <ridge-dir>/<config tag>/<test>/<subset_id>.csv.gz   (ridge.run_tsize)
  TRIPLEX  <triplex-dir>/triplex_tsize_<subset_id>/fold_<test>/preds/*.npz   k = 2, 4
           <triplex-dir>/<k7_tag>/fold_<P>/preds/*.npz                        k = 7
           TRIPLEX PCC is computed here from the preds with the same functions as ridge.

Outputs (in --out)
  tidy.parquet (or tidy.csv.gz without pyarrow)
      model, encoder, context, target, alpha_mode, test_patient, k, subset_id,
      train_patients, gene, pcc, pcc_secavg
      pcc = per-gene PCC over all spots of the test patient, sections pooled (headline);
      pcc_secavg = section-averaged (ridge/score.py) for reference.
  fit_scores.csv      one row per fit: mean / median over genes of pcc, mean of pcc_secavg
  by_k.csv            per model x alpha_mode x test patient x k: mean, sd, min, max over
                      subsets; the k = 7 value; delta and ratio to k = 7
  paired_gap.csv      TRIPLEX minus ridge(cigar), per TRIPLEX run and its IDENTICAL ridge
                      subset: gap of gene means, mean per-gene gap, share of genes TRIPLEX wins
  paired_gap_by_k.csv gap per test patient x k (mean / sd over subsets), change vs k = 7,
                      and an 'ALL' row = mean of the per-patient gaps
  paired_gap_slope.csv  per test patient: OLS slope of the gap against k (subset level)
  k1_matrix.csv       long: model, alpha_mode, train, test, pcc (k = 1 fits)
  k1_matrix/<model>__<alpha_mode>.csv   8 x 8, rows = train patient, cols = test, diag empty
Every comparison is within one test patient; nothing averages raw PCC across patients
except the 'ALL' gap row, which averages within-patient differences.
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd

from ridge import config as rconfig
from ridge.solver import gene_pcc, sections_pcc
from tsize import manifest as tsm

TRIPLEX_MODEL = "triplex"
KEYS = ["model", "alpha_mode", "test_patient"]


def _nanmean(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


# ------------------------------------------------------------------ loading
def load_ridge(ridge_dir, m):
    frames, missing = [], 0
    for cfg in m["ridge_configs"]:
        for s in m["ridge_subsets"]:
            p = os.path.join(ridge_dir, cfg["tag"], s["test"], f"{s['subset_id']}.csv.gz")
            if not os.path.exists(p):
                missing += 1
                continue
            d = pd.read_csv(p)
            d = d.assign(model=cfg["tag"], encoder=cfg["source"], context=cfg["context"],
                         target=cfg["target"], test_patient=s["test"], k=s["k"],
                         subset_id=s["subset_id"], train_patients="".join(s["train"]))
            frames.append(d)
    if missing:
        print(f"[aggregate] ridge: {missing} fits missing (not run yet?)")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def triplex_fold_pcc(fold_dir):
    """(genes, pooled pcc, section-averaged pcc) from one fold's preds/*.npz."""
    files = sorted(glob.glob(os.path.join(fold_dir, "preds", "*.npz")))
    if not files:
        return None
    preds, truths, secs, genes = [], [], [], None
    for f in files:
        z = np.load(f, allow_pickle=True)
        g = [str(x) for x in z["genes"]]
        genes = g if genes is None else genes
        assert g == genes, f"{f}: gene order differs"
        preds.append(z["pred"].astype(np.float64))
        truths.append(z["truth"].astype(np.float64))
        secs += [os.path.basename(f)[:-4]] * len(z["pred"])
    P, T, S = np.vstack(preds), np.vstack(truths), np.array(secs)
    return genes, gene_pcc(P, T), sections_pcc(P, T, S)


def load_triplex(triplex_dir, m):
    runs = list(m["triplex"]["runs"])
    for P in m["patients"]:                       # k = 7 = the existing LOPO run
        tr = [q for q in m["patients"] if q != P]
        runs.append(dict(subset_id=tsm.subset_id(P, tr), test=P, k=7, train=tr,
                         dir=os.path.join(triplex_dir, m["triplex"]["k7_tag"], f"fold_{P}")))
    frames, missing = [], []
    for r in runs:
        d = r.get("dir") or os.path.join(triplex_dir, r["tag"], f"fold_{r['test']}")
        res = triplex_fold_pcc(d)
        if res is None:
            missing.append(r["subset_id"])
            continue
        genes, pooled, sec = res
        frames.append(pd.DataFrame(dict(
            gene=genes, alpha_mode="", pcc=pooled, pcc_secavg=sec, model=TRIPLEX_MODEL,
            encoder="cigar (fine-tuned)", context="triplex", target="cpm_panel_ln",
            test_patient=r["test"], k=r["k"], subset_id=r["subset_id"],
            train_patients="".join(r["train"]))))
    if missing:
        print(f"[aggregate] TRIPLEX: no preds for {missing}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


TIDY_COLS = ["model", "encoder", "context", "target", "alpha_mode", "test_patient", "k",
             "subset_id", "train_patients", "gene", "pcc", "pcc_secavg"]


# ------------------------------------------------------------------ summaries
def fit_scores(tidy):
    g = tidy.groupby(KEYS + ["k", "subset_id", "train_patients"], sort=False)
    out = g.agg(pcc_mean=("pcc", "mean"), pcc_median=("pcc", "median"),
                pcc_secavg_mean=("pcc_secavg", "mean"),
                n_nan_genes=("pcc", lambda s: int(s.isna().sum()))).reset_index()
    return out


def by_k(fits):
    g = fits.groupby(KEYS + ["k"])
    out = g.pcc_mean.agg(n_subsets="count", mean="mean", sd="std", min="min", max="max").reset_index()
    out["pcc_secavg_mean"] = g.pcc_secavg_mean.mean().values
    k7 = fits[fits.k == 7].set_index(KEYS).pcc_mean.rename("pcc_k7")
    out = out.join(k7, on=KEYS)
    out["delta_vs_k7"] = out["mean"] - out.pcc_k7
    out["ratio_vs_k7"] = out["mean"] / out.pcc_k7
    return out.sort_values(KEYS + ["k"]).reset_index(drop=True)


def paired_gap(tidy, m):
    """TRIPLEX minus ridge(cigar), matched on (test patient, subset), per gene."""
    tri = tidy[tidy.model == TRIPLEX_MODEL]
    ridge_models = [c["tag"] for c in m["ridge_configs"] if c["source"] == "cigar"]
    rows = []
    for rm in ridge_models:
        rid = tidy[tidy.model == rm]
        for mode in sorted(rid.alpha_mode.unique()):
            r = rid[rid.alpha_mode == mode]
            j = tri.merge(r[["subset_id", "gene", "pcc"]], on=["subset_id", "gene"],
                          suffixes=("_triplex", "_ridge"))
            for (test, k, sid, trp), d in j.groupby(["test_patient", "k", "subset_id",
                                                     "train_patients"]):
                ok = d.pcc_triplex.notna() & d.pcc_ridge.notna()
                diff = (d.pcc_triplex - d.pcc_ridge)[ok]
                rows.append(dict(ridge_model=rm, alpha_mode=mode, test_patient=test, k=k,
                                 subset_id=sid, train_patients=trp,
                                 triplex=_nanmean(d.pcc_triplex.values),
                                 ridge=_nanmean(d.pcc_ridge.values),
                                 gap=_nanmean(d.pcc_triplex.values) - _nanmean(d.pcc_ridge.values),
                                 gap_per_gene_mean=float(diff.mean()) if len(diff) else np.nan,
                                 frac_genes_triplex_wins=float((diff > 0).mean()) if len(diff) else np.nan,
                                 n_genes=int(ok.sum())))
    return pd.DataFrame(rows)


def paired_gap_summaries(gap, m):
    if gap.empty:
        return gap, gap
    tests = m["triplex"]["test_patients"]
    gap = gap[gap.test_patient.isin(tests)]
    keys = ["ridge_model", "alpha_mode", "test_patient"]
    g = gap.groupby(keys + ["k"])
    byk = g.gap.agg(n_subsets="count", gap_mean="mean", gap_sd="std",
                    gap_min="min", gap_max="max").reset_index()
    byk["triplex_mean"] = g.triplex.mean().values
    byk["ridge_mean"] = g.ridge.mean().values
    byk = byk.join(gap[gap.k == 7].set_index(keys).gap.rename("gap_k7"), on=keys)
    byk["gap_change_vs_k7"] = byk.gap_mean - byk.gap_k7
    # 'ALL': average of the per-patient gaps (each is already a within-patient difference)
    allrows = (byk.groupby(["ridge_model", "alpha_mode", "k"])
               .agg(n_subsets=("n_subsets", "sum"), gap_mean=("gap_mean", "mean"),
                    gap_sd=("gap_mean", "std"), gap_min=("gap_mean", "min"),
                    gap_max=("gap_mean", "max"), triplex_mean=("triplex_mean", "mean"),
                    ridge_mean=("ridge_mean", "mean"), gap_k7=("gap_k7", "mean"),
                    gap_change_vs_k7=("gap_change_vs_k7", "mean"))
               .reset_index().assign(test_patient="ALL"))
    byk = pd.concat([byk, allrows], ignore_index=True).sort_values(keys + ["k"])

    slope_rows = []
    for (rm, mode, test), d in gap.groupby(keys):
        ks = d.k.values.astype(float)
        sl = np.polyfit(ks, d.gap.values, 1)[0] if len(np.unique(ks)) > 1 else np.nan
        slope_rows.append(dict(ridge_model=rm, alpha_mode=mode, test_patient=test,
                               ks=" ".join(str(k) for k in sorted(set(d.k))), n_points=len(d),
                               slope_per_patient=sl,
                               gap_k_min=d[d.k == d.k.min()].gap.mean(),
                               gap_k7=d[d.k == 7].gap.mean() if (d.k == 7).any() else np.nan))
    slope = pd.DataFrame(slope_rows)
    if not slope.empty:
        mean = (slope.groupby(["ridge_model", "alpha_mode"])
                .agg(slope_per_patient=("slope_per_patient", "mean"),
                     gap_k_min=("gap_k_min", "mean"), gap_k7=("gap_k7", "mean"))
                .reset_index().assign(test_patient="ALL"))
        slope = pd.concat([slope, mean], ignore_index=True)
    return byk.reset_index(drop=True), slope


def k1_matrix(fits, patients):
    k1 = fits[fits.k == 1].copy()
    k1["train"] = k1.train_patients
    long = k1[["model", "alpha_mode", "train", "test_patient", "pcc_mean"]].rename(
        columns={"test_patient": "test", "pcc_mean": "pcc"})
    wides = {}
    for (mdl, mode), d in long.groupby(["model", "alpha_mode"]):
        w = d.pivot(index="train", columns="test", values="pcc").reindex(index=patients,
                                                                        columns=patients)
        for p in patients:
            w.loc[p, p] = np.nan
        wides[(mdl, mode)] = w
    return long, wides


# ------------------------------------------------------------------ CLI
def aggregate(ridge_dir, triplex_dir, out, manifest_path=None):
    m = tsm.load(manifest_path)
    os.makedirs(out, exist_ok=True)
    tidy = pd.concat([load_ridge(ridge_dir, m), load_triplex(triplex_dir, m)], ignore_index=True)
    if tidy.empty:
        raise SystemExit("[aggregate] nothing found")
    tidy = tidy[TIDY_COLS]
    try:
        tidy.to_parquet(os.path.join(out, "tidy.parquet"), index=False)
        tidy_name = "tidy.parquet"
    except ImportError:
        tidy.to_csv(os.path.join(out, "tidy.csv.gz"), index=False, compression="gzip")
        tidy_name = "tidy.csv.gz"

    fits = fit_scores(tidy)
    fits.to_csv(os.path.join(out, "fit_scores.csv"), index=False)
    bk = by_k(fits)
    bk.to_csv(os.path.join(out, "by_k.csv"), index=False)
    gap = paired_gap(tidy, m)
    gap.to_csv(os.path.join(out, "paired_gap.csv"), index=False)
    gbk, slope = paired_gap_summaries(gap, m)
    gbk.to_csv(os.path.join(out, "paired_gap_by_k.csv"), index=False)
    slope.to_csv(os.path.join(out, "paired_gap_slope.csv"), index=False)
    long, wides = k1_matrix(fits, m["patients"])
    long.to_csv(os.path.join(out, "k1_matrix.csv"), index=False)
    os.makedirs(os.path.join(out, "k1_matrix"), exist_ok=True)
    for (mdl, mode), w in wides.items():
        w.to_csv(os.path.join(out, "k1_matrix", f"{mdl}__{mode or 'na'}.csv"))

    print(f"[aggregate] {len(tidy):,} rows, {len(fits)} fits -> {out}/{tidy_name}")
    if not gbk.empty:
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            show = gbk[(gbk.alpha_mode == "k7")][["ridge_model", "test_patient", "k", "n_subsets",
                                                  "triplex_mean", "ridge_mean", "gap_mean",
                                                  "gap_change_vs_k7"]]
            print(show.to_string(index=False))
    return dict(tidy=tidy, fits=fits, by_k=bk, gap=gap, gap_by_k=gbk, slope=slope, k1=long)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--ridge-dir", default=os.path.join(rconfig.OUTPUT_DIR, "tsize"))
    ap.add_argument("--triplex-dir", default=os.environ.get("TRIPLEX_OUT", "/workspace/triplex_results"))
    ap.add_argument("--out", default=os.path.join(rconfig.OUTPUT_DIR, "tsize_summary"))
    a = ap.parse_args(argv)
    aggregate(a.ridge_dir, a.triplex_dir, a.out, a.manifest)


if __name__ == "__main__":
    main()
