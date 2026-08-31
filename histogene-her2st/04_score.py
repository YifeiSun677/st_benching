#!/usr/bin/env python
"""Score a finished run.

Headline metric = per-gene PCC computed INSIDE each fold, then averaged across
folds. Pooling every fold's predictions into one matrix before correlating is
contaminated by section-level batch effects on her2st and can flip the sign of
the real signal, so it is reported only as a diagnostic.

    python scripts/04_score.py --tag lopo_833
    python scripts/04_score.py --tag lopo_833 --gene_sets ~/science/results/gene_sets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from htg import config as C   # noqa: E402

MARKERS = ["ERBB2", "GRB7", "ESR1", "PGR", "FASN", "GNAS", "MKI67"]


def pcc_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Column-wise Pearson r between two (n, g) matrices; NaN for zero variance."""
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    num = (A * B).sum(0)
    den = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0))
    out = np.full(A.shape[1], np.nan)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def pcc_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return pcc_cols(A.T, B.T)


def load_run(tag: str, unit: str):
    root = C.OUT_DIR / tag
    folds = sorted(p for p in root.iterdir() if p.is_dir() and (p / "preds").is_dir())
    if not folds:
        raise SystemExit(f"no fold directories under {root}")
    out, genes = [], None
    for fd in folds:
        files = sorted((fd / "preds").glob("*.npz"))
        if not files:
            continue
        P, T, secs = [], [], []
        for f in files:
            z = np.load(f, allow_pickle=True)
            P.append(z["pred"])
            T.append(z["truth"])
            secs.append(f.stem)
            if genes is None:
                genes = [str(g) for g in z["genes"]]
        if unit == "section":
            for p, t, s in zip(P, T, secs):
                out.append({"fold": fd.name, "unit": s, "pred": p, "truth": t})
        else:
            out.append({"fold": fd.name, "unit": fd.name,
                        "pred": np.concatenate(P), "truth": np.concatenate(T)})
    return out, genes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--unit", default="fold", choices=["fold", "section"])
    ap.add_argument("--gene_sets", default=None,
                    help="dir with gene_set_all/hvg/svg/marker.txt")
    args = ap.parse_args()

    units, genes = load_run(args.tag, args.unit)
    G = len(genes)
    print(f"{len(units)} scoring units, {G} genes")

    per_gene, rows = [], []
    for u in units:
        g_r = pcc_cols(u["pred"], u["truth"])
        s_r = pcc_rows(u["pred"], u["truth"])
        base = np.repeat(u["truth"].mean(0, keepdims=True), len(u["truth"]), axis=0)
        sse_model = ((u["pred"] - u["truth"]) ** 2).sum()
        sse_base = ((base - u["truth"]) ** 2).sum()
        per_gene.append(g_r)
        rows.append({
            "fold": u["fold"], "unit": u["unit"], "n_spots": len(u["truth"]),
            "gene_pcc_mean": np.nanmean(g_r), "gene_pcc_median": np.nanmedian(g_r),
            "frac_positive": float(np.nanmean(g_r > 0)),
            "spot_pcc_median": np.nanmedian(s_r),
            "sse_ratio_vs_mean": sse_model / sse_base,
        })
    per_gene = np.vstack(per_gene)              # (n_units, G)
    mean_r = np.nanmean(per_gene, axis=0)

    fold_df = pd.DataFrame(rows)
    gene_df = pd.DataFrame({"gene": genes, "pcc": mean_r,
                            "n_units": np.sum(~np.isnan(per_gene), axis=0)})
    for u, r in zip([x["unit"] for x in units], per_gene):
        gene_df[f"pcc_{u}"] = r

    out = Path("results") / args.tag
    out.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out / "per_fold.csv", index=False)
    gene_df.to_csv(out / "per_gene.csv", index=False)

    summary = {
        "tag": args.tag, "unit": args.unit, "n_units": len(units), "n_genes": G,
        "gene_pcc_median": float(np.nanmedian(mean_r)),
        "gene_pcc_mean": float(np.nanmean(mean_r)),
        "frac_genes_positive": float(np.nanmean(mean_r > 0)),
        "spot_pcc_median": float(fold_df["spot_pcc_median"].median()),
        "sse_ratio_vs_mean": float(fold_df["sse_ratio_vs_mean"].mean()),
        "markers": {m: float(mean_r[genes.index(m)]) for m in MARKERS if m in genes},
    }

    if args.gene_sets:
        gs_dir = Path(args.gene_sets).expanduser()
        strata = {}
        for nm in ("all", "hvg", "svg", "marker"):
            f = gs_dir / f"gene_set_{nm}.txt"
            if not f.exists():
                continue
            keep = [g for g in f.read_text().split() if g in genes]
            idx = [genes.index(g) for g in keep]
            strata[nm] = {"n": len(idx),
                          "pcc_mean": float(np.nanmean(mean_r[idx])),
                          "pcc_median": float(np.nanmedian(mean_r[idx]))}
        summary["gene_sets"] = strata

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(fold_df.to_string(index=False, float_format="%.4f"))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {out}/")


if __name__ == "__main__":
    main()
