"""
Train BLEEP on one fold. Handles both split types through one code path:

  within-patient (as used for the patient-B control):
    --patient B --test_section B1
  leave-one-patient-out:
    --train_sections A1,A2,... --test_sections B1,B2,...
  (run_lopo.sh builds those lists for you)

Protocol, unchanged from the control run:
  * checkpoint = LAST epoch of a fixed pre-declared budget
  * the 80/20 split of training sections is MONITORING ONLY
  * the held-out sections are never touched during training

Reports n_steps in run.json. Step count -- not epoch count -- is what to
match when comparing a LOPO fold against the within-patient control: LOPO
has ~7x more training spots, so equal epochs means ~7x more optimiser
updates.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config_her2st as CFG
import splits
from her2st_dataset import load_panel
from models import CLIPModel
from patch_cache import build_dataset


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


def resolve_split(args):
    if args.train_sections and args.test_sections:
        return (splits.parse_sections(args.train_sections),
                splits.parse_sections(args.test_sections))
    if args.patient and args.test_section:
        return splits.loso_split(args.root, args.patient, args.test_section)
    raise SystemExit("give --train_sections/--test_sections, or "
                     "--patient/--test_section")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", help="patch cache dir from build_patch_cache.py")
    ap.add_argument("--patient")
    ap.add_argument("--test_section")
    ap.add_argument("--train_sections")
    ap.add_argument("--test_sections")
    ap.add_argument("--fold_name", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    panel = load_panel(args.panel)
    if CFG.spot_embedding != len(panel):
        raise SystemExit(f"config spot_embedding={CFG.spot_embedding} != "
                         f"panel {len(panel)}")

    train_sections, test_sections = resolve_split(args)
    print(f"train {len(train_sections)} sections, "
          f"test {len(test_sections)}: {test_sections}")

    full = build_dataset(args, train_sections, panel, is_train=True)
    n_val = int(0.2 * len(full))
    train_ds, val_ds = torch.utils.data.random_split(
        full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    steps_per_epoch = len(train_ds) // args.batch_size
    print(f"train {len(train_ds)} / monitor {len(val_ds)} spots, "
          f"{steps_per_epoch} steps/epoch, "
          f"{steps_per_epoch * args.epochs} total steps")
    if steps_per_epoch < 1:
        raise SystemExit("batch_size exceeds the training set")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

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
    with open(os.path.join(args.out, "run.json"), "w") as fh:
        json.dump({
            "fold_name": args.fold_name or (args.test_section or
                                            ",".join(test_sections)),
            "train_sections": train_sections,
            "test_sections": test_sections,
            "n_train_spots": len(train_ds),
            "n_monitor_spots": len(val_ds),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * args.epochs,
            "n_genes": len(panel),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": CFG.lr, "weight_decay": CFG.weight_decay,
            "temperature": CFG.temperature, "encoder": CFG.model_name,
            "used_cache": bool(args.cache),
            "checkpoint_rule": "last epoch of fixed budget",
            "normalisation": "CPM natural log1p, no Harmony, no stain norm",
            "history": history,
            "wall_seconds": round(time.time() - t0, 1),
        }, fh, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
