"""
Run this FIRST, on CPU. It costs ~2 minutes and catches every failure mode
that would otherwise surface an hour into a GPU run.

    python preflight.py --root /workspace/her2st/data \
                        --panel ../panels/panel_833.txt --patient B
"""
import argparse

import numpy as np
import torch

from her2st_dataset import (Her2stCLIPDataset, load_panel, read_counts,
                            sections_for_patient)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="her2st data dir")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--patient", default="B")
    args = ap.parse_args()

    panel = load_panel(args.panel)
    print(f"panel: {len(panel)} genes, first 3 = {panel[:3]}")

    sections = sections_for_patient(args.root, args.patient)
    print(f"patient {args.patient}: {len(sections)} sections -> {sections}")
    if len(sections) < 3:
        raise SystemExit("need >=3 sections for a within-patient LOSO run")

    # --- panel/count identifier compatibility -----------------------------
    cnt = read_counts(args.root, sections[0])
    overlap = len(set(panel) & set(cnt.columns))
    print(f"\npanel vs {sections[0]} columns: {overlap}/{len(panel)} matched")
    print(f"  example count columns: {list(cnt.columns[:3])}")
    if overlap < 0.5 * len(panel):
        raise SystemExit(
            "FAIL: fewer than half the panel genes match. Your panel file and\n"
            "her2st ST-cnts are probably using different identifier types\n"
            "(ENSG vs HGNC symbol). Fix the panel file before going further."
        )

    # --- full load, one section at a time ---------------------------------
    print("\nLoading all sections:")
    ds = Her2stCLIPDataset(args.root, sections, panel, is_train=True)
    print(f"\ntotal spots: {len(ds)}")

    padded = sum(s.n_padded for s in ds.sections)
    print(f"patches needing edge padding: {padded} "
          f"({100 * padded / len(ds):.1f}%)")
    if padded > 0.05 * len(ds):
        print("  WARNING: >5% padded. Check that pixel_x/pixel_y are in the "
              "same coordinate space as the image you loaded.")

    # --- tensor shape (the bug that kills the original repo) --------------
    item = ds[0]
    img, expr = item["image"], item["reduced_expression"]
    print(f"\nimage tensor: {tuple(img.shape)} dtype={img.dtype}")
    print(f"expr tensor:  {tuple(expr.shape)} dtype={expr.dtype}")
    assert img.shape == (3, 224, 224), f"expected (3,224,224), got {tuple(img.shape)}"
    assert expr.shape == (len(panel),)
    print("  shapes OK")

    # --- sanity on the expression target ----------------------------------
    mat = ds.expression_matrix()
    nz = (mat > 0).mean()
    print(f"\nexpression: min={mat.min():.3f} max={mat.max():.3f} "
          f"mean={mat.mean():.3f}, {100 * nz:.1f}% non-zero")
    per_gene_var = mat.var(axis=0)
    print(f"genes with zero variance across all spots: "
          f"{int((per_gene_var == 0).sum())}/{len(panel)}")

    # --- one forward pass -------------------------------------------------
    print("\nForward pass on a batch of 8...")
    from models import CLIPModel
    model = CLIPModel()
    batch = {
        "image": torch.stack([ds[i]["image"] for i in range(8)]),
        "reduced_expression": torch.stack(
            [ds[i]["reduced_expression"] for i in range(8)]),
    }
    with torch.no_grad():
        loss = model(batch)
    print(f"  loss = {loss.item():.4f}")
    if not np.isfinite(loss.item()):
        raise SystemExit("FAIL: non-finite loss at init")

    print("\nPREFLIGHT PASSED")


if __name__ == "__main__":
    main()
