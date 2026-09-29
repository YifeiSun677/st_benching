#!/usr/bin/env python
"""TCGA-BRCA arm -- per-window aggregates -> patient pseudo-bulk (pre-registered, RUNBOOK design table).

For each model x fold x patient x condition:
    mean_lin_g = sum over the condition's windows of sum_lin_g  /  sum of n_spots        (exact spot mean)
    pb_g       = mean_lin_g / sum_panel(mean_lin) x 1e6                                  (panel CPM, as the bulk)
conditions: frozen = TS + BS pooled (primary) | DX | TS | BS (the last two for the TS-vs-BS ceiling)
fold 'mean' = mean over the 8 folds of pb (the model's headline pseudo-bulk); single folds are kept
so fold spread can be reported.

input : <runs>/fold0<k>_<P>/agg/<SEC>.npz   (external/tcga_common.write_agg)
output: <out>/pseudobulk_<model>.tsv.gz      long-ish: model, fold, patient, cond, n_spots, n_windows,
                                             then the 833 panel genes
usage : python tcga/pseudobulk.py --runs /workspace/runs/tcga_stnet --out /workspace/results/tcga
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

COND = {"frozen": ("TS", "BS"), "DX": ("DX",), "TS": ("TS",), "BS": ("BS",)}


def load(runs: Path) -> tuple[pd.DataFrame, np.ndarray, list[str], str]:
    rows, S, genes, model = [], [], None, None
    for f in sorted(glob.glob(str(runs / "fold0*_*" / "agg" / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        g = [str(x) for x in z["genes"]]
        genes = genes or g
        assert g == genes, f"{f}: gene order differs"
        model = model or str(z["model"])
        rows.append(dict(fold=int(z["fold"]), section=str(z["section"]), patient=str(z["patient"]),
                         kind=str(z["kind"]), n_spots=int(z["n_spots"])))
        S.append(z["sum_lin"])
    if not rows:
        raise SystemExit(f"no agg files under {runs}")
    return pd.DataFrame(rows), np.vstack(S), genes, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="/workspace/results/tcga")
    ap.add_argument("--windows", default=os.environ.get("TCGA_WINDOWS", "/workspace/ext/tcga/meta/windows.tsv"))
    a = ap.parse_args()
    meta, S, genes, model = load(Path(a.runs))

    # completeness: every expected window x fold must be there, or the pseudo-bulk is silently partial
    w = pd.read_csv(a.windows, sep="\t")
    folds = sorted(meta.fold.unique())
    have = set(zip(meta.fold, meta.section))
    missing = [(f, s) for f in folds for s in w.section if (f, s) not in have]
    print(f"{model}: {len(meta)} window x fold aggregates, folds {folds}, "
          f"{meta.section.nunique()}/{len(w)} windows; missing {len(missing)}")
    if missing:
        pats = sorted({s.split('_')[0] for _, s in missing})
        print(f"  WARNING incomplete patients (dropped from the output): {len(pats)} e.g. {pats[:5]}")
        meta_ok = ~meta.patient.isin(pats)
        meta, S = meta[meta_ok.values].reset_index(drop=True), S[meta_ok.values]

    out = []
    for (fold, patient), g in meta.groupby(["fold", "patient"]):
        for cond, kinds in COND.items():
            m = g.kind.isin(kinds).values
            if not m.any():
                continue
            idx = g.index[m]
            mean_lin = S[idx].sum(0) / meta.n_spots[idx].sum()
            pb = mean_lin / mean_lin.sum() * 1e6
            out.append(dict(model=model, fold=str(fold), patient=patient, cond=cond,
                            n_spots=int(meta.n_spots[idx].sum()), n_windows=int(m.sum()), **dict(zip(genes, pb))))
    df = pd.DataFrame(out)
    avg = (df.groupby(["patient", "cond"])[genes].mean().reset_index()
           .merge(df[df.fold == str(folds[0])][["patient", "cond", "n_spots", "n_windows"]], on=["patient", "cond"]))
    avg.insert(0, "fold", "mean")
    avg.insert(0, "model", model)
    df = pd.concat([avg[df.columns], df], ignore_index=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    p = Path(a.out) / f"pseudobulk_{model}.tsv.gz"
    df.to_csv(p, sep="\t", index=False, compression="gzip", float_format="%.6g")
    print(f"wrote {p}: {df[df.fold == 'mean'].patient.nunique()} patients, "
          f"{df[df.fold == 'mean'].groupby('cond').size().to_dict()}")


if __name__ == "__main__":
    main()
