"""
ridge/summarize.py -- one table of every scored run under RIDGE_OUT.

    python -m ridge.summarize                 # prints + writes $RIDGE_OUT/summary.csv

For each run: target, model, source, context, folds found, alpha per fold
(and whether any fold hit the grid edge), then the headline scores on all
sections and on the 8 annotated sections (if scored with --sections annotated).
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

from . import config


def _load(p):
    with open(p) as f:
        return json.load(f)


def summarize(out_root=None):
    out_root = out_root or config.OUTPUT_DIR
    rows = []
    for run in sorted(glob.glob(os.path.join(out_root, "*", "run_config.json"))):
        d = os.path.dirname(run)
        cfg = _load(run)
        folds = sorted(glob.glob(os.path.join(d, "fold_*", "fold.json")))
        fj = [_load(f) for f in folds]
        alphas = [f.get("alpha") for f in fj if "alpha" in f]
        row = dict(tag=os.path.basename(d), model=cfg.get("model"), source=cfg.get("source") or "",
                   context=cfg.get("context") if cfg.get("model") == "ridge" else "",
                   target=cfg.get("target"), n_folds=len(folds),
                   alpha_min=min(alphas) if alphas else None,
                   alpha_max=max(alphas) if alphas else None,
                   grid_edge=any(f.get("alpha_at_grid_edge") for f in fj))
        for suffix, name in (("", "all"), ("_annotated", "ann")):
            p = os.path.join(d, f"headline{suffix}.json")
            if os.path.exists(p):
                h = _load(p)
                row[f"pcc_{name}"] = h.get("pcc_mean")
                row[f"mae_{name}"] = h.get("mae_mean")
                for gs in ("hvg", "svg", "marker"):
                    if f"pcc_{gs}" in h and name == "all":
                        row[f"pcc_{gs}"] = h[f"pcc_{gs}"]["mean"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[summarize] no runs under {out_root}")
        return df
    df = df.sort_values(["target", "model", "source", "context"])
    df.to_csv(os.path.join(out_root, "summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))
    print(f"\n[summarize] -> {os.path.join(out_root, 'summary.csv')}")
    if (df.n_folds != len(config.PATIENTS)).any():
        print("[summarize] WARNING: some runs do not have 8 folds:",
              df.loc[df.n_folds != len(config.PATIENTS), "tag"].tolist())
    if df.grid_edge.any():
        print("[summarize] WARNING: alpha hit the grid edge in:",
              df.loc[df.grid_edge, "tag"].tolist())
    return df


if __name__ == "__main__":
    summarize()
