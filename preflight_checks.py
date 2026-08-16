"""
Pre-flight checks for Hist2ST on her2st with the common 833-gene panel.

Run this BEFORE any timing or training run. Every check prints PASS / FAIL /
WARN and the script exits non-zero if anything FAILs, so it can gate the
launch script.

    python preflight_checks.py \
        --repo /workspace/Hist2ST \
        --data /workspace/her2st/data \
        --panel /workspace/panels/panel_833.txt \
        --stnet-genes /workspace/panels/stnet_run2_gene_order.txt

Checks
------
A  learning rate + model construction   -> must be 1e-5, model must build with n_genes=833
B  gene panel                           -> dataset must expose exactly the 833 panel genes,
                                           in a known order, matched against the ST-Net run-2 order
C  adjacency graph                      -> mean degree ~= k, coordinates on the array scale
                                           (tens), not the pixel scale (thousands)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

FAILS: list[str] = []
WARNS: list[str] = []


def ok(msg):
    print(f"  PASS  {msg}")


def fail(msg):
    print(f"  FAIL  {msg}")
    FAILS.append(msg)


def warn(msg):
    print(f"  WARN  {msg}")
    WARNS.append(msg)


def header(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def load_panel(path: Path) -> list[str]:
    if path.suffix == ".npy":
        return [str(g) for g in np.load(path, allow_pickle=True)]
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# Check A
# --------------------------------------------------------------------------- #
def check_a(repo: Path, n_genes: int, expected_lr: float):
    header("CHECK A  -- model construction and learning rate")
    sys.path.insert(0, str(repo))
    try:
        from HIST2ST import Hist2ST  # type: ignore
    except Exception as e:
        fail(f"cannot import Hist2ST from {repo}: {e}")
        return None

    import inspect

    sig = inspect.signature(Hist2ST.__init__)
    defaults = {
        k: v.default for k, v in sig.parameters.items() if v.default is not inspect._empty
    }
    print("  constructor defaults:")
    for k in sorted(defaults):
        print(f"      {k:<16} {defaults[k]}")

    lr_default = defaults.get("learning_rate", defaults.get("lr"))
    if lr_default is None:
        warn("no learning_rate/lr in the constructor signature -- set it explicitly")
    elif abs(float(lr_default) - expected_lr) < 1e-12:
        ok(f"repo default learning_rate = {lr_default} (matches the intended {expected_lr})")
    else:
        warn(
            f"repo default learning_rate = {lr_default}, intended {expected_lr}. "
            "This is fine, but it is now a CHOICE and must be justified in methods, "
            "not inherited from the paper."
        )

    try:
        model = Hist2ST(n_genes=n_genes)
    except Exception as e:
        fail(f"Hist2ST(n_genes={n_genes}) failed to construct: {e}")
        return None

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok(f"model built: {n_params/1e6:.2f} M params ({n_train/1e6:.2f} M trainable)")

    # the output head must actually be 833-wide
    out_dims = {
        name: tuple(p.shape)
        for name, p in model.named_parameters()
        if n_genes in tuple(p.shape)
    }
    if out_dims:
        ok(f"{len(out_dims)} parameter tensors carry the gene dimension {n_genes}")
        for name in list(out_dims)[:6]:
            print(f"      {name:<40} {out_dims[name]}")
    else:
        fail(
            f"no parameter tensor has dimension {n_genes} -- n_genes was probably "
            "ignored and the head is still the default width"
        )
    return model


# --------------------------------------------------------------------------- #
# Check B
# --------------------------------------------------------------------------- #
def check_b(repo: Path, data: Path, panel: list[str], stnet_order: list[str] | None,
            fold: int):
    header("CHECK B  -- gene panel identity and order")
    sys.path.insert(0, str(repo))
    try:
        from dataset import ViT_HER2ST  # type: ignore
    except Exception as e:
        fail(f"cannot import ViT_HER2ST from {repo}/dataset.py: {e}")
        return None

    print(f"  panel file: {len(panel)} genes, first 5 = {panel[:5]}")
    if len(set(panel)) != len(panel):
        fail("panel file contains duplicate gene names")

    try:
        ds = ViT_HER2ST(train=True, gene_list=list(panel), fold=fold)
    except TypeError as e:
        fail(
            f"ViT_HER2ST does not accept gene_list the way assumed ({e}). "
            "Inspect dataset.py: the loader hard-codes data/her_hvg_cut_1000.npy in "
            "some revisions and must be patched to read the 833 panel."
        )
        return None
    except Exception as e:
        fail(f"ViT_HER2ST construction failed: {e}")
        return None

    # the attribute holding the resolved gene list differs across revisions
    resolved = None
    for attr in ("gene_set", "gene_list", "genes"):
        if hasattr(ds, attr):
            v = getattr(ds, attr)
            if v is not None and len(v) > 0:
                resolved = [str(g) for g in v]
                print(f"  resolved gene list from ds.{attr}")
                break
    if resolved is None:
        warn("could not find the resolved gene list attribute on the dataset")
    else:
        if len(resolved) != len(panel):
            fail(
                f"dataset exposes {len(resolved)} genes but the panel has {len(panel)} "
                "-- genes were silently dropped (absent from the her2st count matrix?)"
            )
            missing = sorted(set(panel) - set(resolved))
            print(f"      missing ({len(missing)}): {missing[:20]}")
        elif resolved != list(panel):
            warn(
                "same genes but a DIFFERENT ORDER than the panel file. Not fatal, but "
                "record this order -- prediction matrices must be aligned by name, "
                "never by position."
            )
        else:
            ok("dataset gene list matches the panel file exactly, in order")

        if stnet_order is not None:
            if resolved == stnet_order:
                ok("gene order identical to the ST-Net run-2 output -- matrices align by position")
            elif set(resolved) == set(stnet_order):
                warn(
                    "same gene SET as ST-Net run 2 but a different ORDER. "
                    "Align by gene name when pooling across models."
                )
            else:
                fail("gene set differs from ST-Net run 2 -- the two models are not comparable")

    # section / spot inventory
    try:
        n_sections = len(ds)
        spot_counts = []
        for i in range(n_sections):
            item = ds[i]
            exp = None
            for t in item:
                if hasattr(t, "shape") and t.ndim == 2 and t.shape[-1] == len(panel):
                    exp = t
                    break
            spot_counts.append(int(exp.shape[0]) if exp is not None else -1)
        print(f"  sections in this split: {n_sections}")
        print(f"  spots per section: min {min(spot_counts)}, median "
              f"{int(np.median(spot_counts))}, max {max(spot_counts)}, "
              f"total {sum(spot_counts)}")
        if min(spot_counts) < 0:
            warn("could not identify the expression tensor in some sections")
        else:
            ok("expression tensors have the expected gene dimension")
    except Exception as e:
        warn(f"could not inventory sections: {e}")
    return ds


# --------------------------------------------------------------------------- #
# Check C
# --------------------------------------------------------------------------- #
def check_c(ds, k: int):
    header("CHECK C  -- adjacency graph and coordinate scale")
    if ds is None:
        fail("skipped: dataset unavailable")
        return

    bad_scale, bad_degree = 0, 0
    for i in range(min(len(ds), 8)):
        item = ds[i]
        coords, adj = None, None
        for t in item:
            if not hasattr(t, "shape"):
                continue
            arr = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
            if arr.ndim == 2 and arr.shape[1] == 2 and coords is None:
                coords = arr
            elif arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 2:
                adj = arr
        if coords is None or adj is None:
            warn(f"section {i}: could not locate coords and/or adjacency in the batch")
            continue

        span = coords.max(0) - coords.min(0)
        deg = adj.sum(1)
        deg_mean = float(deg.mean())
        # self-loops shift the degree by 1 in some revisions
        plausible = abs(deg_mean - k) < 1.0 or abs(deg_mean - (k + 1)) < 1.0

        print(
            f"  section {i:>2}: coord span {span.astype(int)}  "
            f"mean degree {deg_mean:.2f}  isolated nodes {(deg == 0).sum()}"
        )
        if span.max() > 1000:
            bad_scale += 1
        if not plausible:
            bad_degree += 1

    if bad_scale:
        fail(
            f"{bad_scale} section(s) have a coordinate span > 1000 -- these look like PIXEL "
            "coordinates. calcADJ will still build a graph and nothing will error, but the "
            "k-NN neighbourhood will not be the spatial neighbourhood and the GNN branch "
            "learns noise. Feed array indices (x, y) instead."
        )
    else:
        ok("coordinate spans are on the array scale")

    if bad_degree:
        fail(f"{bad_degree} section(s) have a mean degree far from k={k} -- check prune/k settings")
    else:
        ok(f"mean degree consistent with k={k}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path, help="path to the Hist2ST clone")
    ap.add_argument("--data", required=True, type=Path, help="path to her2st/data")
    ap.add_argument("--panel", required=True, type=Path, help="833-gene panel file (.txt or .npy)")
    ap.add_argument("--stnet-genes", type=Path, default=None,
                    help="gene order used by the ST-Net 833 run, for cross-model alignment")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    panel = load_panel(args.panel)
    stnet_order = load_panel(args.stnet_genes) if args.stnet_genes else None

    check_a(args.repo, n_genes=len(panel), expected_lr=args.lr)
    ds = check_b(args.repo, args.data, panel, stnet_order, args.fold)
    check_c(ds, k=args.k)

    header("SUMMARY")
    print(f"  {len(FAILS)} FAIL, {len(WARNS)} WARN")
    for m in FAILS:
        print(f"    FAIL: {m}")
    for m in WARNS:
        print(f"    WARN: {m}")
    if FAILS:
        print("\n  Do not launch the timing run until the FAILs are resolved.")
        sys.exit(1)
    print("\n  Clear to run profile_hist2st.py")


if __name__ == "__main__":
    main()
