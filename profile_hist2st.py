"""
Timing + efficiency profiling for Hist2ST on her2st (833-gene panel).

Runs a small number of epochs on ONE fold with a manual training loop (not the
Lightning Trainer) so every gradient step can be timed individually, then a
separate inference pass. Produces the four efficiency-slide numbers and the
per-epoch cost needed to set the epoch ceiling.

    python profile_hist2st.py \
        --repo /workspace/Hist2ST \
        --panel /workspace/panels/panel_833.txt \
        --fold 0 --warmup-epochs 1 --measure-epochs 2 \
        --out results/efficiency/hist2st_833_train.json

Then, for the bake ablation (only if the default is too slow):
    python profile_hist2st.py ... --bake 0 --out results/efficiency/hist2st_833_bake0.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from profiling import ResourceMonitor, StepTimer, write_report


def load_panel(path: Path) -> list[str]:
    if path.suffix == ".npy":
        return [str(g) for g in np.load(path, allow_pickle=True)]
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def to_device(batch, dev):
    if torch.is_tensor(batch):
        return batch.to(dev, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(b, dev) for b in batch)
    return batch


def n_spots_of(batch) -> int:
    """Number of spots in a section-batch. Patch tensor is [1, N, C, H, W]."""
    for t in batch:
        if torch.is_tensor(t) and t.dim() == 5:
            return int(t.shape[1])
    for t in batch:
        if torch.is_tensor(t) and t.dim() == 3 and t.shape[0] == 1:
            return int(t.shape[1])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bake", type=int, default=5)
    ap.add_argument("--lamb", type=float, default=0.5)
    ap.add_argument("--zinb", type=float, default=0.25)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--measure-epochs", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--sample-interval", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("results/efficiency/hist2st_833_train.json"))
    ap.add_argument("--csv", type=Path, default=Path("results/efficiency/efficiency.csv"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo))
    from HIST2ST import Hist2ST  # type: ignore
    from dataset import ViT_HER2ST  # type: ignore

    panel = load_panel(args.panel)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={dev}  genes={len(panel)}  fold={args.fold}  bake={args.bake}")

    train_ds = ViT_HER2ST(train=True, gene_list=list(panel), fold=args.fold)
    test_ds = ViT_HER2ST(train=False, gene_list=list(panel), fold=args.fold)
    # one section per batch -- this IS Hist2ST's batching, do not change it
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"[setup] {len(train_ds)} training sections, {len(test_ds)} held-out section(s)")

    model = Hist2ST(
        n_genes=len(panel),
        learning_rate=args.lr,
        bake=args.bake,
        lamb=args.lamb,
        zinb=args.zinb,
    ).to(dev)
    opt_cfg = model.configure_optimizers()
    opt = opt_cfg[0][0] if isinstance(opt_cfg, tuple) else (
        opt_cfg["optimizer"] if isinstance(opt_cfg, dict) else opt_cfg
    )
    if isinstance(opt, (list, tuple)):
        opt = opt[0]
    print(f"[setup] optimizer={type(opt).__name__} lr={opt.param_groups[0]['lr']}")

    mon = ResourceMonitor(interval=args.sample_interval)
    train_timer = StepTimer()
    epoch_seconds: list[float] = []

    model.train()
    with mon:
        # ---- warm-up (cudnn autotune, page cache, worker spin-up) ---------- #
        for ep in range(args.warmup_epochs):
            t0 = time.perf_counter()
            for batch in train_dl:
                batch = to_device(batch, dev)
                loss = model.training_step(batch, 0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize() if dev.type == "cuda" else None
            print(f"[warmup] epoch {ep}: {time.perf_counter() - t0:.1f} s (not measured)")

        # ---- measured window ---------------------------------------------- #
        mon.reset_window()
        for ep in range(args.measure_epochs):
            t_ep = time.perf_counter()
            for batch in train_dl:
                batch = to_device(batch, dev)
                ns = n_spots_of(batch)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                loss = model.training_step(batch, 0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                train_timer.add(time.perf_counter() - t0, n_spots=ns)
            epoch_seconds.append(time.perf_counter() - t_ep)
            print(f"[measure] epoch {ep}: {epoch_seconds[-1]:.1f} s")

    train_stats = {**mon.summary(), **train_timer.summary(prefix="train_")}
    sec_per_epoch = float(np.mean(epoch_seconds)) if epoch_seconds else None

    # ---- inference pass (separate monitor, separate numbers) --------------- #
    inf_timer = StepTimer()
    mon_inf = ResourceMonitor(interval=args.sample_interval)
    model.eval()
    with mon_inf, torch.no_grad():
        mon_inf.reset_window()
        for batch in test_dl:
            batch = to_device(batch, dev)
            ns = n_spots_of(batch)
            tensors = [t for t in batch if torch.is_tensor(t)]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                _ = model(tensors[0], tensors[1], tensors[3])  # patch, center, adj
            except Exception:
                _ = model(*tensors[:3])
            if dev.type == "cuda":
                torch.cuda.synchronize()
            inf_timer.add(time.perf_counter() - t0, n_spots=ns)
    inf_stats = {
        **{f"infer_{k}": v for k, v in mon_inf.summary().items()},
        **inf_timer.summary(prefix="infer_"),
    }

    payload = {
        "model": "Hist2ST",
        "panel": f"{len(panel)}-gene common panel",
        "fold": args.fold,
        "lr": args.lr,
        "bake": args.bake,
        "lamb": args.lamb,
        "zinb": args.zinb,
        "batching": "1 section per step (method default)",
        "num_workers": args.num_workers,
        "n_train_sections": len(train_ds),
        "steps_per_epoch": len(train_ds),
        "sec_per_epoch_mean": round(sec_per_epoch, 2) if sec_per_epoch else None,
        **train_stats,
        **inf_stats,
    }

    # ---- projections that decide the ceiling ------------------------------- #
    if sec_per_epoch:
        print("\n" + "=" * 66)
        print("PROJECTED COST  (8 outer folds x [3 inner + 1 refit] = 32 trainings)")
        print("=" * 66)
        print(f"{'ceiling':>8} {'per training':>14} {'full LOPO':>14}")
        for ceiling in (100, 150, 200, 250, 350):
            per = sec_per_epoch * ceiling
            print(f"{ceiling:>8} {per/3600:>12.1f} h {per*32/3600:>12.1f} h")
        payload["projected_lopo_hours"] = {
            str(c): round(sec_per_epoch * c * 32 / 3600, 1)
            for c in (100, 150, 200, 250, 350)
        }
        print("=" * 66)

    write_report(args.out, payload, also_csv=args.csv)


if __name__ == "__main__":
    main()
