"""
ridge/score.py -- score a run with EXACTLY the benchmark's headline definition
(triplex/score.py): per section, per gene PCC (NaN when either side is constant);
nanmean over the held-out patient's sections -> per-gene fold vector; mean over
folds per gene (NaN-skipping); headline = mean over genes.

Additions for the baselines:
  mae         per-gene mean |pred - truth|, averaged the same way. Needed because a
              constant prediction (train_mean) has NaN PCC by definition.
  n_nan_genes genes whose fold-level PCC is NaN (all of them for train_mean).
  --sections  restrict scoring to some sections, e.g. the 8 annotated ones, so a
              region_mean run and any other run can be compared on the same spots.

    python -m ridge.score --tag ridge_uni_hest112_local_raw_ln
    python -m ridge.score --tag train_mean_cpm_panel_ln --sections annotated
    python -m ridge.score --run-dir /path/to/other_port/<tag> --sections annotated
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

from . import config, data
from .solver import gene_pcc

ANNOTATED = ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]


def _fold_dirs(run_dir):
    """fold_A ... (this port, TRIPLEX) or fold00_A ... (STFlow) -> {patient: dir}."""
    out = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "fold*"))):
        m = re.search(r"fold\w*?_([A-H])$", os.path.basename(d))
        if m and os.path.isdir(os.path.join(d, "preds")):
            out[m.group(1)] = d
    return out


def _nanmean(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis)


def score_dir(run_dir, sections=None, out_dir=None, quiet=False):
    folds = _fold_dirs(run_dir)
    if not folds:
        raise FileNotFoundError(f"no fold*/preds under {run_dir}")
    genes, pcc_by_fold, mae_by_fold, summary = None, {}, {}, []
    for P, fdir in folds.items():
        pccs, maes, sse_r, used = [], [], [], []
        for f in sorted(glob.glob(os.path.join(fdir, "preds", "*.npz"))):
            sec = os.path.basename(f)[:-4]
            if sections is not None and sec not in sections:
                continue
            z = np.load(f, allow_pickle=True)
            g = [str(x) for x in z["genes"]]
            genes = g if genes is None else genes
            assert g == genes, f"{f}: gene order differs from other sections"
            pred, truth = z["pred"].astype(np.float64), z["truth"].astype(np.float64)
            pccs.append(gene_pcc(pred, truth))
            maes.append(np.abs(pred - truth).mean(0))
            base = ((truth - truth.mean(0)) ** 2).sum(0)       # section's own true mean (oracle)
            with np.errstate(divide="ignore", invalid="ignore"):
                sse_r.append(((pred - truth) ** 2).sum(0) / base)
            used.append(sec)
        if not used:
            continue
        pcc = _nanmean(np.vstack(pccs))
        pcc_by_fold[P], mae_by_fold[P] = pcc, np.mean(maes, 0)
        sser = np.vstack(sse_r)
        summary.append(dict(patient=P, sections=" ".join(used),
                            pcc_mean=float(np.nanmean(pcc)) if np.isfinite(pcc).any() else np.nan,
                            pcc_median=float(np.nanmedian(pcc)) if np.isfinite(pcc).any() else np.nan,
                            frac_pos=float(np.mean(pcc > 0)),
                            mae_mean=float(np.mean(mae_by_fold[P])),
                            median_sse_ratio=float(np.nanmedian(_nanmean(sser))),
                            n_nan_genes=int(np.isnan(pcc).sum())))
    if not pcc_by_fold:
        raise ValueError(f"no sections left to score in {run_dir} (sections={sections})")

    pg = pd.DataFrame(pcc_by_fold, index=genes)
    pg.index.name = "gene"
    mg = pd.DataFrame(mae_by_fold, index=genes)
    mg.index.name = "gene"
    across = pg.mean(1)                                  # per gene over folds, NaN-skipping
    mae_across = mg.mean(1)
    headline = dict(run_dir=run_dir, patients=list(pcc_by_fold),
                    sections=sorted(sections) if sections else "all",
                    pcc_mean=float(across.mean()) if across.notna().any() else None,
                    pcc_median=float(across.median()) if across.notna().any() else None,
                    frac_pos=float((across > 0).mean()),
                    n_nan_genes=int(across.isna().sum()),
                    mae_mean=float(mae_across.mean()),
                    markers={m: (float(across[m]) if pd.notna(across.get(m)) else None)
                             for m in config.MARKERS if m in across.index})
    for name, mask in data.load_gene_sets(list(pg.index)).items():
        sub = across[mask]
        headline[f"pcc_{name}"] = dict(n=int(mask.sum()),
                                       mean=float(sub.mean()) if sub.notna().any() else None,
                                       median=float(sub.median()) if sub.notna().any() else None,
                                       mae=float(mae_across[mask].mean()))

    out_dir = out_dir or run_dir
    suffix = "" if sections is None else "_" + ("annotated" if sorted(sections) == sorted(ANNOTATED)
                                                 else "subset")
    os.makedirs(out_dir, exist_ok=True)
    pg.to_csv(os.path.join(out_dir, f"per_gene_pcc_by_fold{suffix}.csv"))
    mg.to_csv(os.path.join(out_dir, f"per_gene_mae_by_fold{suffix}.csv"))
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, f"per_fold_summary{suffix}.csv"), index=False)
    with open(os.path.join(out_dir, f"headline{suffix}.json"), "w") as f:
        json.dump(headline, f, indent=2)
    if not quiet:
        print(json.dumps({k: v for k, v in headline.items() if k != "run_dir"}, indent=2))
        print(f"[score] -> {out_dir}/headline{suffix}.json")
    return headline


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tag", help="a run under RIDGE_OUT")
    g.add_argument("--run-dir", help="any run dir with fold*/preds/*.npz (e.g. another port)")
    ap.add_argument("--sections", nargs="+",
                    help="section names, or 'annotated' for the 8 pathologist-labelled sections")
    ap.add_argument("--out-dir", help="where to write scores (default: the run dir)")
    args = ap.parse_args(argv)
    run_dir = args.run_dir or os.path.join(config.OUTPUT_DIR, args.tag)
    sections = None
    if args.sections:
        sections = ANNOTATED if args.sections == ["annotated"] else args.sections
    return score_dir(run_dir, sections, args.out_dir)


if __name__ == "__main__":
    main()
