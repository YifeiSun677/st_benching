"""
Extract and freeze the gene order used by the ST-Net 833-gene run.

This order is the alignment key for every model in the benchmark: once a second
model writes an 833-column prediction matrix, the two can only be pooled safely
if the column order is either identical or reconciled by name. Freezing it into
a plain text file in the repo means that reconciliation never depends on
re-opening a large .npz that lives outside git.

    # single file
    python extract_gene_order.py \
        --npz /workspace/runs/stnet_833/A1_25.npz \
        --out panels/stnet_run2_gene_order.txt

    # all folds, with a cross-fold consistency check (recommended)
    python extract_gene_order.py \
        --npz-dir /workspace/runs/stnet_833 --glob "*_25.npz" \
        --panel panels/panel_833.txt \
        --out panels/stnet_run2_gene_order.txt

If the .npz turns out not to carry gene names at all, the script says so and
lists what it does contain, so the order can be recovered from the run's own
panel file instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def load_panel(path: Path) -> list[str]:
    if path.suffix == ".npy":
        return [str(g) for g in np.load(path, allow_pickle=True)]
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def describe(npz) -> None:
    print("  contents:")
    for k in npz.files:
        a = npz[k]
        kind = getattr(a, "dtype", "?")
        shape = getattr(a, "shape", "?")
        print(f"      {k:<24} shape={shape} dtype={kind}")


def find_gene_names(npz, expect_n: int | None = None) -> tuple[list[str], str] | None:
    """Locate the 1-D array of gene identifiers inside a prediction .npz.

    Strategy: prefer conventional key names; otherwise take any 1-D array of
    strings whose length matches the gene dimension of a 2-D matrix in the file.
    """
    preferred = ["gene_names", "genes", "gene", "gene_list", "columns",
                 "var_names", "gene_symbols", "gene_ids"]

    # infer the gene dimension from the largest 2-D array (spots x genes)
    if expect_n is None:
        mats = [npz[k] for k in npz.files
                if getattr(npz[k], "ndim", 0) == 2]
        if mats:
            biggest = max(mats, key=lambda a: a.size)
            expect_n = int(biggest.shape[1])
            print(f"  inferred gene dimension = {expect_n} "
                  f"(from a {biggest.shape} matrix)")

    def is_stringy(a) -> bool:
        return a.dtype.kind in ("U", "S", "O")

    for key in preferred:
        if key in npz.files:
            a = npz[key]
            if a.ndim == 1 and is_stringy(a):
                return [str(g) for g in a], key

    for key in npz.files:
        a = npz[key]
        if a.ndim == 1 and is_stringy(a) and (expect_n is None or len(a) == expect_n):
            return [str(g) for g in a], key

    return None


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", type=Path, help="a single prediction .npz")
    src.add_argument("--npz-dir", type=Path, help="directory of prediction .npz files")
    ap.add_argument("--glob", default="*.npz", help="pattern within --npz-dir")
    ap.add_argument("--panel", type=Path, default=None,
                    help="833-gene panel file, for a set-membership cross-check")
    ap.add_argument("--expect", type=int, default=833, help="expected gene count")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = [args.npz] if args.npz else sorted(args.npz_dir.glob(args.glob))
    if not files:
        sys.exit(f"no files matched {args.npz_dir}/{args.glob}")
    print(f"[read] {len(files)} file(s)")

    orders: dict[str, list[str]] = {}
    for f in files:
        npz = np.load(f, allow_pickle=True)
        print(f"\n{f.name}")
        found = find_gene_names(npz, expect_n=args.expect)
        if found is None:
            print("  no 1-D string array found -- this .npz carries no gene names.")
            describe(npz)
            print(
                "\n  Recover the order from the run's own panel file instead: ST-Net "
                "writes columns in the order it received them, so the panel file passed "
                "to that run IS the order. Copy it to --out and verify its length is "
                f"{args.expect}."
            )
            sys.exit(2)
        genes, key = found
        print(f"  gene names from key '{key}': {len(genes)} entries, "
              f"first 3 = {genes[:3]}, last 3 = {genes[-3:]}")
        orders[f.name] = genes

    # --- cross-fold consistency ------------------------------------------- #
    reference = orders[files[0].name]
    mismatched = [n for n, g in orders.items() if g != reference]
    print()
    if mismatched:
        print(f"  MISMATCH: {len(mismatched)} file(s) differ from {files[0].name}")
        for n in mismatched[:5]:
            g = orders[n]
            if set(g) == set(reference):
                print(f"    {n}: same set, different order -- align by NAME downstream")
            else:
                only_ref = sorted(set(reference) - set(g))[:5]
                only_g = sorted(set(g) - set(reference))[:5]
                print(f"    {n}: different set. only in ref: {only_ref} | only here: {only_g}")
        sys.exit(3)
    elif len(files) > 1:
        print(f"  PASS  gene order identical across all {len(files)} folds")

    # --- sanity checks ------------------------------------------------------ #
    if len(reference) != args.expect:
        print(f"  WARN  {len(reference)} genes, expected {args.expect}")
    else:
        print(f"  PASS  {len(reference)} genes as expected")

    if len(set(reference)) != len(reference):
        dup = sorted({g for g in reference if reference.count(g) > 1})
        print(f"  WARN  duplicate gene names: {dup[:10]}")

    if args.panel:
        panel = load_panel(args.panel)
        if set(panel) == set(reference):
            if panel == reference:
                print("  PASS  identical to the panel file, same order")
            else:
                print("  NOTE  same gene set as the panel file but a different order. "
                      "The ORDER WRITTEN HERE is authoritative for the ST-Net matrices; "
                      "align other models by name against it.")
        else:
            miss = sorted(set(panel) - set(reference))
            extra = sorted(set(reference) - set(panel))
            print(f"  WARN  differs from the panel file. "
                  f"in panel not in run ({len(miss)}): {miss[:10]} | "
                  f"in run not in panel ({len(extra)}): {extra[:10]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(reference) + "\n")
    print(f"\n[write] {args.out}  ({len(reference)} genes)")
    print("  Commit this file. It is the alignment key for the whole benchmark.")


if __name__ == "__main__":
    main()
