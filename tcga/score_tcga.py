#!/usr/bin/env python
"""TCGA-BRCA arm -- pre-registered scoring (RUNBOOK design table).  Pure numpy/pandas.

Within each stratum (HER2pos primary; LumA, LumB, Basal), for condition frozen (primary) and DX:
  r_g        Spearman across patients between pseudo-bulk (fold mean, panel CPM) and bulk panel CPM
  headline   median r_g over each gene set (all / hvg / svg / marker); also frac(r_g > 0)
  p_perm     one-sided patient-label permutation within the stratum, B = 1000:
             (1 + #{null median >= observed}) / (B + 1)
  CI         2.5 / 97.5 % of the median over B = 1000 patient bootstraps
  folds      median r_g (all genes) of each single fold -> min / max
Ceiling (patients with both TS and BS, per stratum): median over genes of Spearman(TS pred, BS pred)
across patients = how reproducible the model's pseudo-bulk is between two faces of the same block.

No pooled analysis across strata (pre-registered).

inputs : pseudobulk_<model>.tsv.gz (tcga/pseudobulk.py), meta/rna/rna_panel_cpm.tsv, meta/cohort.csv,
         results/gene_sets/gene_set_<name>.txt
outputs: <out>/scores_<model>.tsv, <out>/per_gene_<model>.tsv.gz
usage  : python tcga/score_tcga.py --pb /workspace/results/tcga/pseudobulk_stnet.tsv.gz
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
# genes with no variance in a stratum (e.g. undetected in every patient) give all-NaN columns: expected
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
SETS = ("all", "hvg", "svg", "marker")
STRATA = ("HER2pos", "LumA", "LumB", "Basal")


def zrank(X: np.ndarray) -> np.ndarray:
    """Column-wise ranks (ties averaged), standardised; constant columns -> NaN."""
    R = rankdata(X, axis=0)
    R = R - R.mean(0)
    sd = R.std(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return R / np.where(sd > 0, sd, np.nan)


def spearman_cols(X, Y) -> np.ndarray:
    return np.nanmean(zrank(X) * zrank(Y), axis=0) if len(X) > 2 else np.full(X.shape[1], np.nan)


def summarise(r: np.ndarray, idx: dict) -> dict:
    return {s: float(np.nanmedian(r[i])) for s, i in idx.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb", required=True)
    ap.add_argument("--meta", default="/workspace/ext/tcga/meta")
    ap.add_argument("--gene-sets", default=str(REPO / "results" / "gene_sets"))
    ap.add_argument("--out", default=None, help="default: next to --pb")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260929)
    a = ap.parse_args()
    pb = pd.read_csv(a.pb, sep="\t")
    model = pb.model.iloc[0]
    meta = Path(a.meta)
    rna = pd.read_csv(meta / "rna" / "rna_panel_cpm.tsv", sep="\t", index_col=0)
    coh = pd.read_csv(meta / "cohort.csv", index_col=0)
    genes = [c for c in pb.columns if c not in ("model", "fold", "patient", "cond", "n_spots", "n_windows")]
    rna = rna[genes]
    gs = {s: [g.strip() for g in open(Path(a.gene_sets) / f"gene_set_{s}.txt") if g.strip()] for s in SETS}
    idx = {s: np.array([genes.index(g) for g in v if g in genes]) for s, v in gs.items()}
    rng = np.random.default_rng(a.seed)
    out = Path(a.out) if a.out else Path(a.pb).parent

    rows, per_gene = [], []
    for cond in ("frozen", "DX"):
        for st in STRATA:
            pats = sorted(set(coh.index[(coh.role == "main") & (coh.stratum == st)])
                          & set(pb.loc[(pb.fold == "mean") & (pb.cond == cond), "patient"]))
            if len(pats) < 4:
                continue
            X = pb[(pb.fold == "mean") & (pb.cond == cond)].set_index("patient").loc[pats, genes].to_numpy(float)
            Y = rna.loc[pats].to_numpy(float)
            r = spearman_cols(X, Y)
            obs = summarise(r, idx)
            zx, zy = zrank(X), zrank(Y)
            null = {s: np.empty(a.B) for s in SETS}
            boot = {s: np.empty(a.B) for s in SETS}
            n = len(pats)
            for b in range(a.B):
                rp = np.nanmean(zx[rng.permutation(n)] * zy, axis=0)
                ib = rng.integers(0, n, n)
                rb = spearman_cols(X[ib], Y[ib])
                for s in SETS:
                    null[s][b] = np.nanmedian(rp[idx[s]])
                    boot[s][b] = np.nanmedian(rb[idx[s]])
            fold_r = []
            for f in sorted(set(pb.fold) - {"mean"}):
                Xf = pb[(pb.fold == f) & (pb.cond == cond)].set_index("patient").reindex(pats)[genes].to_numpy(float)
                fold_r.append(np.nanmedian(spearman_cols(Xf, Y)))
            for s in SETS:
                rows.append(dict(model=model, cond=cond, stratum=st, gene_set=s, n_patients=n,
                                 n_genes=int(np.isfinite(r[idx[s]]).sum()), median_r=round(obs[s], 4),
                                 ci_lo=round(float(np.nanpercentile(boot[s], 2.5)), 4),
                                 ci_hi=round(float(np.nanpercentile(boot[s], 97.5)), 4),
                                 frac_pos=round(float(np.nanmean(r[idx[s]] > 0)), 3),
                                 p_perm=round((1 + int((null[s] >= obs[s]).sum())) / (a.B + 1), 4),
                                 fold_min=round(float(np.min(fold_r)), 4) if s == "all" else np.nan,
                                 fold_max=round(float(np.max(fold_r)), 4) if s == "all" else np.nan))
            per_gene.append(pd.DataFrame(dict(model=model, cond=cond, stratum=st, gene=genes, r=r)))

        # TS vs BS reproducibility ceiling (only once, independent of cond)
    for st in STRATA:
        m = pb[(pb.fold == "mean") & pb.cond.isin(["TS", "BS"])]
        both = sorted(set(m[m.cond == "TS"].patient) & set(m[m.cond == "BS"].patient)
                      & set(coh.index[(coh.role == "main") & (coh.stratum == st)]))
        if len(both) < 4:
            rows.append(dict(model=model, cond="TSvsBS", stratum=st, gene_set="all", n_patients=len(both)))
            continue
        Xt = m[m.cond == "TS"].set_index("patient").loc[both, genes].to_numpy(float)
        Xb = m[m.cond == "BS"].set_index("patient").loc[both, genes].to_numpy(float)
        Y = rna.loc[both].to_numpy(float)
        for s in SETS:
            rows.append(dict(model=model, cond="TSvsBS", stratum=st, gene_set=s, n_patients=len(both),
                             median_r=round(float(np.nanmedian(spearman_cols(Xt, Xb)[idx[s]])), 4),
                             r_TS_rna=round(float(np.nanmedian(spearman_cols(Xt, Y)[idx[s]])), 4),
                             r_BS_rna=round(float(np.nanmedian(spearman_cols(Xb, Y)[idx[s]])), 4)))

    res = pd.DataFrame(rows)
    res.to_csv(out / f"scores_{model}.tsv", sep="\t", index=False)
    pd.concat(per_gene).to_csv(out / f"per_gene_{model}.tsv.gz", sep="\t", index=False, compression="gzip",
                               float_format="%.4f")
    show = res[res.gene_set.isin(["all", "marker"])]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
