"""
Train BLEEP on one within-patient fold.

Replaces BLEEP_main.py, whose main() reads SLURM_LOCALID / SLURM_NODEID and
hard-crashes outside SLURM. Single-GPU, no DDP.

Two deliberate protocol choices, both to be stated in methods:

  1. Checkpoint = LAST epoch of a fixed pre-declared budget (default 4,
     BLEEP's own argparse default). BLEEP saves the best-val-loss
     checkpoint instead; we do not, because this benchmark scores every
     model at the last epoch of its own declared budget.
  2. The 80/20 split of the training sections is used for MONITORING ONLY.
     Nothing is selected on it. The held-out section is never touched
     during training.

    python train_bleep.py --root /workspace/her2st/data \
        --panel ../panels/panel_833.txt --patient B --test_section B1 \
        --out /workspace/runs/bleep_patientB/B1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config_her2st as CFG
from her2st_dataset import Her2stCLIPDataset, load_panel, sections_for_patient
from models import CLIPModel


class AvgMeter:
    def __init__(self):
        self.sum = self.count = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


def run_epoch(model, loader, device, optimizer=None):
    meter = AvgMeter()
    train = optimizer is not None
    model.train(train)
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if train:
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = model(batch)
        meter.update(loss.item(), batch["image"].size(0))
    return meter.avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--patient", default="B")
    ap.add_argument("--test_section", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    panel = load_panel(args.panel)
    all_sections = sections_for_patient(args.root, args.patient)
    if args.test_section not in all_sections:
        raise SystemExit(f"{args.test_section} not in {all_sections}")
    train_sections = [s for s in all_sections if s != args.test_section]

    print(f"fold: test={args.test_section}  train={train_sections}")
    if CFG.spot_embedding != len(panel):
        raise SystemExit(
            f"config_her2st.spot_embedding={CFG.spot_embedding} but panel has "
            f"{len(panel)} genes. Edit config_her2st.py."
        )

    full = Her2stCLIPDataset(args.root, train_sections, panel, is_train=True)
    n_val = int(0.2 * len(full))
    train_ds, val_ds = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"train spots {len(train_ds)}  /  monitor spots {len(val_ds)}")
    if len(train_ds) < args.batch_size:
        raise SystemExit("batch_size exceeds training set; contrastive "
                         "learning needs the in-batch negatives -- get more "
                         "sections rather than shrinking the batch")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            drop_last=False)

    model = CLIPModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr,
                                  weight_decay=CFG.weight_decay)

    history, t0 = [], time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device, None)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"epoch {epoch}/{args.epochs}  train {tr:.4f}  monitor {va:.4f}")

    torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
    meta = {
        "patient": args.patient,
        "test_section": args.test_section,
        "train_sections": train_sections,
        "n_train_spots": len(train_ds),
        "n_monitor_spots": len(val_ds),
        "n_genes": len(panel),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": CFG.lr,
        "weight_decay": CFG.weight_decay,
        "temperature": CFG.temperature,
        "encoder": CFG.model_name,
        "checkpoint_rule": "last epoch of fixed budget",
        "normalisation": "CPM natural log1p, no Harmony, no stain norm",
        "history": history,
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "run.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved -> {args.out}  ({meta['wall_seconds']}s)")


if __name__ == "__main__":
    main()
