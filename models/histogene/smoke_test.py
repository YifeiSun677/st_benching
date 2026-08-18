#!/usr/bin/env python
"""
Run this before anything else. Three checks, in order of how likely they are
to fail:

  1. pytorch-lightning 1.9.5 can actually drive torch 2.8 on GPU.
     (import succeeding proves nothing -- PL 1.9.5 predates torch 2.x by years.)
  2. torch.load still works for PL checkpoints. torch >= 2.6 flipped
     weights_only to True by default and PL 1.9.5 does not pass the argument,
     so load_from_checkpoint raises at inference time, not training time.
  3. One real HisToGene epoch on fold 0, reporting per-epoch time and peak
     GPU memory. Use these numbers to budget the full run.

Usage:
    python smoke_test.py --panel panels/panel_833.txt
"""
import argparse
import functools
import time

import torch

torch.load = functools.partial(torch.load, weights_only=False)


def check_trainer():
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset

    class M(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(4, 2)

        def training_step(self, b, i):
            return torch.nn.functional.mse_loss(self.l(b[0]), b[1])

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), 1e-3)

    dl = DataLoader(TensorDataset(torch.randn(8, 4), torch.randn(8, 2)), batch_size=2)
    pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, logger=False,
               enable_checkpointing=False, enable_progress_bar=False,
               enable_model_summary=False).fit(M(), dl)
    print(f"[1/3] OK  torch {torch.__version__}  pl {pl.__version__}  "
          f"gpu {torch.cuda.get_device_name(0)}")


def check_checkpoint_roundtrip(tmp="/tmp/_htg_smoke.ckpt"):
    import pytorch_lightning as pl
    from vis_model import HisToGene

    m = HisToGene(n_layers=1, n_genes=10, dim=64, patch_size=112, n_pos=64)
    torch.save({"state_dict": m.state_dict(),
                "hyper_parameters": {},
                "pytorch-lightning_version": pl.__version__}, tmp)
    HisToGene.load_from_checkpoint(tmp, n_layers=1, n_genes=10, dim=64,
                                   patch_size=112, n_pos=64)
    print("[2/3] OK  load_from_checkpoint round-trips under torch.load patch")


def check_one_epoch(panel, cache, fold, precision):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from dataset_fast import CachedHER2ST, load_panel
    from vis_model import HisToGene

    genes = load_panel(panel)

    t = time.time()
    ds = CachedHER2ST(train=True, fold=fold, gene_list=genes, cache=cache)
    build_s = time.time() - t
    print(f"      {ds.summary()}")

    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=True)
    model = HisToGene(n_layers=8, n_genes=len(genes), learning_rate=1e-5,
                      patch_size=112, dim=1024, n_pos=64, dropout=0.1)

    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    pl.Trainer(accelerator="gpu", devices=1, max_epochs=1, precision=precision,
               logger=False, enable_checkpointing=False,
               enable_model_summary=False).fit(model, loader)
    epoch_s = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"[3/3] OK  dataset build {build_s:.0f}s | epoch {epoch_s:.1f}s | "
          f"peak GPU {peak:.1f} GB")
    print(f"\n      projected: 100 epochs = {(build_s + 100*epoch_s)/60:.0f} min/fold, "
          f"8 folds serial = {8*(build_s + 100*epoch_s)/3600:.1f} h")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="panels/panel_833.txt")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--precision", default="32")
    a = ap.parse_args()

    check_trainer()
    check_checkpoint_roundtrip()
    check_one_epoch(a.panel, a.cache, a.fold, a.precision)
