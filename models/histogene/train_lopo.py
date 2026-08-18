#!/usr/bin/env python
"""
Train HisToGene on her2st, leave-one-patient-out.

    python train_lopo.py --fold 0 --panel panels/panel_833.txt

Hyperparameters follow the repo's tutorial.ipynb (n_layers=8, lr=1e-5,
100 epochs, no validation split, no early stopping). Note the README's own
default is n_layers=4; the tutorial uses 8. Whichever you pick, record it --
it is the single biggest cost knob in this model.

One "batch" is one whole section, so an epoch is only ~31 gradient steps.
100 epochs = ~3100 steps total. That is by design, not a misconfiguration.
"""
import argparse
import functools
import json
import os
import time

import torch

torch.load = functools.partial(torch.load, weights_only=False)

import pytorch_lightning as pl  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from dataset_fast import CachedHER2ST, PATIENTS, load_panel  # noqa: E402
from vis_model import HisToGene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, help="0-7 -> hold out patient A-H")
    ap.add_argument("--panel", default="panels/panel_833.txt")
    ap.add_argument("--tag", default="htg_her2st_833_lopo")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--precision", default="32", help="32 or bf16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pl.seed_everything(a.seed, workers=True)
    os.makedirs("model", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    genes = load_panel(a.panel)
    ds = CachedHER2ST(train=True, fold=a.fold, gene_list=genes, cache=a.cache)
    print(ds.summary(), flush=True)

    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=True)
    model = HisToGene(n_layers=a.n_layers, n_genes=len(genes), learning_rate=a.lr,
                      patch_size=112, dim=a.dim, n_pos=64, dropout=a.dropout)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=a.epochs,
                         precision=a.precision, enable_checkpointing=False,
                         default_root_dir=f"logs/{a.tag}_{a.fold}")
    trainer.fit(model, loader)
    mins = (time.time() - t0) / 60

    ckpt = f"model/last_train_{a.tag}_{a.fold}.ckpt"
    trainer.save_checkpoint(ckpt)

    meta = dict(vars(a), n_genes=len(genes), held_out=PATIENTS[a.fold],
                train_sections=ds.names, minutes=round(mins, 1),
                peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                torch=torch.__version__, pl=pl.__version__)
    with open(f"logs/{a.tag}_{a.fold}_run.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nfold {a.fold} ({PATIENTS[a.fold]}) done in {mins:.1f} min -> {ckpt}")


if __name__ == "__main__":
    main()
