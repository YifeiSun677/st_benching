"""
Score a TRIPLEX LOPO run on the same footing as the rest of the benchmark:
per-fold, per-gene PCC averaged across folds (NOT pooled -- pooling is
contaminated by section batch effects), each section scored against its own
stored truth inside the npz (pred and truth are row-aligned, so no join).

Emits, under OUTPUT_DIR/<tag>/:
    per_gene_pcc_by_fold.csv   gene x patient PCC
    per_fold_summary.csv       per-patient pcc mean/median, frac_pos,
                               median SSE ratio, frac_beat_baseline, sd ratio
    headline.json              cross-fold means + marker genes + gene-set means
"""
import os
import glob
import json
import numpy as np
import pandas as pd

from . import config


def _pcc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def _section_metrics(pred, truth):
    """Per-gene PCC, plus SSE ratio vs each gene's per-section mean."""
    n_genes = pred.shape[1]
    pcc = np.array([_pcc(pred[:, g], truth[:, g]) for g in range(n_genes)])
    base = np.repeat(truth.mean(0, keepdims=True), len(truth), axis=0)
    sse_model = ((pred - truth) ** 2).sum(0)
    sse_base = ((base - truth) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sse_ratio = sse_model / sse_base
    sd_ratio = pred.std(0) / (truth.std(0) + 1e-12)
    return pcc, sse_ratio, sd_ratio


def _load_fold(tag, patient):
    """Average per-gene metrics across the held-out patient's sections."""
    fdir = os.path.join(config.OUTPUT_DIR, tag, f"fold_{patient}", "preds")
    files = sorted(glob.glob(os.path.join(fdir, "*.npz")))
    pcc_s, sser_s, sdr_s = [], [], []
    genes = None
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        genes = z["genes"].astype(str)
        pcc, sser, sdr = _section_metrics(z["pred"], z["truth"])
        pcc_s.append(pcc); sser_s.append(sser); sdr_s.append(sdr)
    return (genes,
            np.nanmean(pcc_s, 0), np.nanmedian(sser_s, 0), np.nanmean(sdr_s, 0))


def _gene_sets():
    out = {}
    for name in ("all", "hvg", "svg", "marker"):
        fp = os.path.join(config.GENE_SETS, f"gene_set_{name}.txt")
        if os.path.exists(fp):
            with open(fp) as f:
                out[name] = [ln.strip() for ln in f if ln.strip()]
    return out


def score_run(tag):
    patients = []
    genes = None
    pcc_by_fold, summary = {}, []
    for p in config.PATIENTS:
        if not os.path.isdir(os.path.join(config.OUTPUT_DIR, tag, f"fold_{p}")):
            continue
        g, pcc, sser, sdr = _load_fold(tag, p)
        genes = g if genes is None else genes
        patients.append(p)
        pcc_by_fold[p] = pcc
        summary.append(dict(
            patient=p,
            pcc_mean=float(np.nanmean(pcc)),
            pcc_median=float(np.nanmedian(pcc)),
            frac_pos=float(np.mean(pcc > 0)),
            median_sse_ratio=float(np.nanmedian(sser)),
            frac_beat_baseline=float(np.mean(sser < 1.0)),
            sd_ratio_median=float(np.nanmedian(sdr)),
        ))

    if not patients:
        print(f"[score] no folds found under {config.OUTPUT_DIR}/{tag}")
        return

    out_dir = os.path.join(config.OUTPUT_DIR, tag)
    pg = pd.DataFrame(pcc_by_fold, index=genes)
    pg.index.name = "gene"
    pg.to_csv(os.path.join(out_dir, "per_gene_pcc_by_fold.csv"))
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "per_fold_summary.csv"),
                                 index=False)

    across = pg.mean(1)                          # per-gene, averaged over folds
    headline = dict(
        tag=tag, patients=patients,
        pcc_mean=float(across.mean()), pcc_median=float(across.median()),
        frac_pos=float((across > 0).mean()),
        markers={m: float(across[m]) for m in config.MARKERS if m in across.index},
    )
    for name, gs in _gene_sets().items():
        sub = across[across.index.isin(gs)]
        if len(sub):
            headline[f"pcc_{name}"] = dict(n=int(len(sub)),
                                           mean=float(sub.mean()),
                                           median=float(sub.median()))
    with open(os.path.join(out_dir, "headline.json"), "w") as f:
        json.dump(headline, f, indent=2)

    print(json.dumps(headline, indent=2))
    print(f"[score] wrote per_gene_pcc_by_fold.csv, per_fold_summary.csv, "
          f"headline.json -> {out_dir}")
    return headline


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="triplex_lopo_833")
    score_run(ap.parse_args().tag)
