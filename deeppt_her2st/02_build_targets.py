#!/usr/bin/env python
"""
Step 2 — 833-gene target matrix, zero-filled, log10(CP10K + 1).

Benchmark-wide rules applied here:
  * gene order FIXED to panel_833.txt across every section and every fold
  * genes absent from a section's ST-cnts are ZERO-FILLED, not intersected
    (her2st count files only carry genes detected in that section)
  * transform log10(CP10K + 1), matching HisToGene. log10 vs ln differ by a
    constant factor, so PCC is IDENTICAL either way -- only MSE / R^2 /
    baseline-gain are affected. DeepPT vs HisToGene PCC is directly comparable.

CP10K is computed over the FULL count matrix (all detected genes), then the
panel is subset. Normalising after subsetting would make library size depend
on panel choice and break the 250/785/833 panel comparison.

Output:
    targets/<sec>.npy      float32 [n_spots, 833]  (row-aligned to features)
    targets/genes.txt      the 833 gene ids, in order
    targets/coverage.csv   per-gene per-section presence, for the write-up
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import her2st_io as io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--her2st", required=True)
    ap.add_argument("--panel", required=True, help="panels/panel_833.txt")
    ap.add_argument("--features", required=True, help="output dir of step 1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    panel = [l.strip() for l in open(args.panel) if l.strip()]
    print(f"[panel] {len(panel)} genes from {args.panel}")

    sections = io.list_sections(args.her2st)
    coverage = {}

    for sec in sections:
        # Row order MUST match step 1 exactly.
        spots = pd.read_csv(os.path.join(args.features, f"{sec}_spots.csv"))
        ids = spots["spot_id"].astype(str).tolist()

        cnt = io.load_counts(args.her2st, sec).loc[ids]
        lib = cnt.sum(axis=1).values.astype(np.float64)
        lib[lib == 0] = 1.0

        present = [g for g in panel if g in cnt.columns]
        coverage[sec] = len(present)

        mat = np.zeros((len(ids), len(panel)), dtype=np.float64)
        if present:
            idx = [panel.index(g) for g in present]
            sub = cnt[present].values.astype(np.float64)
            mat[:, idx] = sub / lib[:, None] * 1e4
        mat = np.log10(mat + 1.0)

        np.save(os.path.join(args.out, f"{sec}.npy"), mat.astype(np.float32))
        print(f"  {sec}: {len(ids)} spots, {len(present)}/{len(panel)} panel genes present")

    with open(os.path.join(args.out, "genes.txt"), "w") as fh:
        fh.write("\n".join(panel) + "\n")
    pd.Series(coverage, name="panel_genes_present").to_csv(
        os.path.join(args.out, "coverage.csv"))

    lo = min(coverage.values())
    print(f"\n[done] worst section carries {lo}/{len(panel)} panel genes.")
    if lo < 0.5 * len(panel):
        print("  !! <50% overlap -- check the panel id type (symbol vs ENSG) "
              "against ST-cnts column names before going further.")


if __name__ == "__main__":
    main()
