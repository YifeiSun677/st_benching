#!/usr/bin/env python
"""
Export predictions for one held-out patient.

    python predict_lopo.py --fold 0 --panel panels/panel_833.txt

Deliberately does NOT use predict.py's model_predict(). That function calls
preds.squeeze() on the first batch only and then torch.cat's later batches
onto it, which breaks whenever the test set has more than one section -- which
is always true under LOPO (patients A-D have six sections each).

Also skips super-resolution entirely: sr=True tiles a 56 px grid over the whole
slide, giving ~15k tokens into full self-attention, which OOMs even on 24 GB
and is not needed for the benchmark.

Output: processed/<tag>_fold<k>.npz with
    pred     (n_spots, n_genes)  float32
    truth    (n_spots, n_genes)  float32   library-size normalized + log, same as training target
    coords   (n_spots, 2)        pixel coords
    section  (n_spots,)          section label per spot, e.g. 'A1'
    genes    (n_genes,)
"""
import argparse
import functools

import numpy as np
import torch

torch.load = functools.partial(torch.load, weights_only=False)

from dataset_fast import CachedHER2ST, PATIENTS, load_panel  # noqa: E402
from vis_model import HisToGene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--panel", default="panels/panel_833.txt")
    ap.add_argument("--tag", default="htg_her2st_833_lopo")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="processed")
    a = ap.parse_args()

    import os
    os.makedirs(a.out, exist_ok=True)
    genes = load_panel(a.panel)
    dev = torch.device("cuda")

    # HisToGene.__init__ has save_hyperparameters() commented out, so every
    # constructor argument must be re-supplied here and must match training
    # exactly -- otherwise the weights load into a differently shaped model.
    model = HisToGene.load_from_checkpoint(
        f"model/last_train_{a.tag}_{a.fold}.ckpt",
        n_layers=a.n_layers, n_genes=len(genes), learning_rate=a.lr,
        patch_size=112, dim=a.dim, n_pos=64)
    model.eval().to(dev)

    ds = CachedHER2ST(train=False, fold=a.fold, gene_list=genes, cache=a.cache)
    print(ds.summary(), flush=True)

    preds, truths, coords, labels = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            patches, loc, exp, centers = ds[i]
            p = model(patches.unsqueeze(0).to(dev), loc.unsqueeze(0).to(dev))
            p = p.squeeze(0).cpu().numpy()
            preds.append(p)
            truths.append(exp.numpy())
            coords.append(centers.numpy())
            labels.append(np.repeat(ds.names[i], len(p)))
            print(f"  {ds.names[i]}: {p.shape}", flush=True)

    out = f"{a.out}/{a.tag}_fold{a.fold}.npz"
    np.savez_compressed(
        out,
        pred=np.concatenate(preds).astype(np.float32),
        truth=np.concatenate(truths).astype(np.float32),
        coords=np.concatenate(coords),
        section=np.concatenate(labels),
        genes=np.array(genes),
    )
    n = sum(len(p) for p in preds)
    print(f"\nfold {a.fold} ({PATIENTS[a.fold]}): {n} spots x {len(genes)} genes -> {out}")


if __name__ == "__main__":
    main()
