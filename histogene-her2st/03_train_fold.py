#!/usr/bin/env python
"""Train HisToGene on one fold and write predictions for the held-out sections.

    python scripts/03_train_fold.py --cv patient --fold 0 --tag lopo_833
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from htg import cache, config as C, her2st                      # noqa: E402
from htg.dataset import HER2STSections                          # noqa: E402

sys.path.insert(0, str(C.HISTOGENE_REPO))
import pytorch_lightning as pl                                  # noqa: E402
from vis_model import HisToGene                                 # noqa: E402


class EpochLoss(pl.Callback):
    """Records mean training loss per epoch (the upstream run.json has no curve)."""

    def __init__(self):
        self.rows, self._buf, self._t0 = [], [], None

    def on_train_epoch_start(self, trainer, pl_module):
        self._buf, self._t0 = [], time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self._buf.append(float(loss.detach()))

    def on_train_epoch_end(self, trainer, pl_module):
        self.rows.append({
            "epoch": trainer.current_epoch + 1,
            "train_loss": float(np.mean(self._buf)) if self._buf else float("nan"),
            "seconds": round(time.time() - self._t0, 2),
        })
        r = self.rows[-1]
        print(f"  epoch {r['epoch']:3d}  loss {r['train_loss']:.4f}  {r['seconds']:.1f}s",
              flush=True)


@torch.no_grad()
def predict(model, sections, panel, device):
    model.eval().to(device)
    ds = HER2STSections(sections, panel, train=False)
    out = {}
    for i in range(len(ds)):
        patches, positions, exps, centers = ds[i]
        pred = model(patches.unsqueeze(0).to(device), positions.unsqueeze(0).to(device))
        out[ds.sections[i]] = {
            "pred": pred.squeeze(0).float().cpu().numpy(),
            "truth": exps.numpy(),
            "centers": centers.numpy(),
            "spot_id": np.array(ds.coords[ds.sections[i]]["spot_id"], dtype=object),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default=C.CV, choices=["patient", "section"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--n_layers", type=int, default=C.N_LAYERS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--panel", default=str(C.PANEL_FILE))
    ap.add_argument("--save_ckpt", action="store_true")
    args = ap.parse_args()

    pl.seed_everything(C.SEED, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    panel = her2st.load_panel(args.panel)
    fold = her2st.folds(args.cv)[args.fold]
    run_dir = C.OUT_DIR / args.tag / f"fold{args.fold:02d}_{fold['name']}"
    (run_dir / "preds").mkdir(parents=True, exist_ok=True)

    print(f"== {args.tag} | cv={args.cv} fold={args.fold} ({fold['name']}) ==")
    print(f"   train sections: {len(fold['train'])}  test sections: {fold['test']}")
    print(f"   genes: {len(panel)}  epochs: {args.epochs}  lr: {args.lr}")

    train_ds = HER2STSections(fold["train"], panel, train=True)
    print(f"   train spots: {train_ds.n_spots()}")
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)

    model = HisToGene(patch_size=C.PATCH_SIZE, n_layers=args.n_layers,
                      n_genes=len(panel), dim=C.DIM, learning_rate=args.lr,
                      dropout=C.DROPOUT, n_pos=C.N_POS)

    cb = EpochLoss()
    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    trainer = pl.Trainer(
        accelerator="gpu" if gpu else "cpu", devices=1,
        max_epochs=args.epochs, callbacks=[cb],
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, log_every_n_steps=1, precision="32-true",
    )
    t0 = time.time()
    trainer.fit(model, train_loader)
    train_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if gpu else 0.0

    with open(run_dir / "loss_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "seconds"])
        w.writeheader()
        w.writerows(cb.rows)

    if args.save_ckpt:
        trainer.save_checkpoint(run_dir / "last.ckpt")

    device = torch.device("cuda" if gpu else "cpu")
    t1 = time.time()
    preds = predict(model, fold["test"], panel, device)
    pred_s = time.time() - t1

    for sec, d in preds.items():
        np.savez_compressed(run_dir / "preds" / f"{sec}.npz",
                            pred=d["pred"], truth=d["truth"],
                            centers=d["centers"], spot_id=d["spot_id"],
                            genes=np.array(panel, dtype=object))

    (run_dir / "run.json").write_text(json.dumps({
        "tag": args.tag, "cv": args.cv, "fold": args.fold, "held_out": fold["name"],
        "train_sections": fold["train"], "test_sections": fold["test"],
        "n_genes": len(panel), "epochs": args.epochs, "lr": args.lr,
        "n_layers": args.n_layers, "dim": C.DIM, "dropout": C.DROPOUT,
        "patch_size": C.PATCH_SIZE, "n_pos": C.N_POS, "seed": C.SEED,
        "target": "log10(CP10K+1) over panel columns; missing genes zero-filled",
        "scored_at": "last epoch (no validation-based selection)",
        "train_seconds": round(train_s, 1), "predict_seconds": round(pred_s, 1),
        "sec_per_epoch": round(train_s / max(args.epochs, 1), 2),
        "peak_gpu_gb": round(peak_gb, 2),
        "final_train_loss": cb.rows[-1]["train_loss"] if cb.rows else None,
    }, indent=2))

    print(f"-- done in {train_s/60:.1f} min "
          f"({train_s/max(args.epochs,1):.1f} s/epoch), peak GPU {peak_gb:.2f} GB")
    print(f"-- wrote {run_dir}")


if __name__ == "__main__":
    main()
