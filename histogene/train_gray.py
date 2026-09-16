"""HisToGene colour ablation, one fold, three arms.

    python -m histogene.train_gray --fold 0 --tag htg_gray_lopo_833

Arms (same names/semantics as the BLEEP patient-B ablation):
    arm1_colour     colour train  -> colour test        (reference; gray_mode "none")
    arm2_graytest   colour train  -> grayscale test     (perturbation control; "query")
    arm3_graytrain  gray   train  -> grayscale test     (retrained; "all")

arm1 and arm2 come from ONE colour model trained in this process: it is scored
on colour test patches and again on grayscale test patches, so the only
difference between arm1 and arm2 is the test input. No checkpoint round-trip,
no chance of pairing the wrong weights. Because of that, arm1 and arm2 are
always (re)written together.

arm3 is a separate model with the same seed, epochs, lr and fold, trained and
scored on grayscale patches, so arm3 vs arm1 differs only in the pixels.

Budget: the main-table config (100 epochs, lr 1e-5, last epoch scored).

Output
    /workspace/runs/<tag>/<arm>/foldNN_<P>/preds/<section>.npz
                                         /loss_curve.csv   (arm1, arm3)
                                         /run.json
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

from . import config as C, her2st
from .dataset import HER2STSections
from .gray_bridge import (assert_canonical, channel_spread_max, gray_provenance,
                          gray_section)

sys.path.insert(0, str(C.HISTOGENE_REPO))
import pytorch_lightning as pl                                  # noqa: E402
from vis_model import HisToGene                                 # noqa: E402

ARMS = {
    1: {"dir": "arm1_colour",    "gray_train": False, "gray_test": False, "mode": "none"},
    2: {"dir": "arm2_graytest",  "gray_train": False, "gray_test": True,  "mode": "query"},
    3: {"dir": "arm3_graytrain", "gray_train": True,  "gray_test": True,  "mode": "all"},
}


class EpochLoss(pl.Callback):
    """Mean training loss per epoch."""

    def __init__(self, label: str):
        self.label = label
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
        if r["epoch"] == 1 or r["epoch"] % 10 == 0:
            print(f"  [{self.label}] epoch {r['epoch']:4d}  loss {r['train_loss']:.4f}  "
                  f"{r['seconds']:.2f}s", flush=True)


# --------------------------------------------------------------- checks -----
def verify_input(ds: HER2STSections, expect_gray: bool, panel) -> dict:
    """Prove what the model actually receives, on the first item of `ds`.

    1. channel spread of the flattened float tensor: 0 iff grayscale.
    2. for gray datasets: the item equals canonical-gray(colour cache) exactly,
       i.e. the transform is the one in bleep/gray.py and nothing else changed.
    3. shape is still (n, 37632) and scale is still 0-255.
    """
    item = ds[0]
    x = item[0].numpy()
    spread = channel_spread_max(x)
    info = {
        "section_checked": ds.sections[0],
        "shape": list(x.shape),
        "max_value": float(x.max()),
        "channel_spread_max": spread,
    }
    if x.shape[1] != C.PATCH_DIM:
        raise RuntimeError(f"patch dim {x.shape[1]} != {C.PATCH_DIM}")
    if x.max() <= 1.5:
        raise RuntimeError("patches look 0-1 scaled; HisToGene uses raw 0-255")
    if expect_gray:
        if spread != 0:
            raise RuntimeError(f"gray dataset but channel spread = {spread}")
        colour = HER2STSections([ds.sections[0]], panel, train=True, gray=False)
        ref = gray_section(colour.raw_patches(0)).transpose(0, 2, 1, 3)
        ref = ref.reshape(ref.shape[0], -1).astype(np.float32)
        same = bool(np.array_equal(ref, x))
        info["equals_canonical_gray_of_cache"] = same
        if not same:
            raise RuntimeError("gray item != canonical gray(cache); transform path differs")
    else:
        if spread == 0:
            raise RuntimeError("colour dataset but every pixel has R==G==B")
    return info


# ---------------------------------------------------------------- train -----
def train_model(train_secs, panel, gray: bool, args, label: str):
    pl.seed_everything(C.SEED, workers=True)        # same init + order for both models
    ds = HER2STSections(train_secs, panel, train=True, gray=gray)
    check = verify_input(ds, expect_gray=gray, panel=panel)
    print(f"  [{label}] input check: spread={check['channel_spread_max']:.0f} "
          f"max={check['max_value']:.0f} shape={tuple(check['shape'])}", flush=True)

    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    model = HisToGene(patch_size=C.PATCH_SIZE, n_layers=args.n_layers,
                      n_genes=len(panel), dim=C.DIM, learning_rate=args.lr,
                      dropout=C.DROPOUT, n_pos=C.N_POS)
    in_dim = model.patch_embedding.in_features
    if in_dim != C.PATCH_DIM:
        raise RuntimeError(f"patch_embedding in_features {in_dim} != {C.PATCH_DIM}")

    cb = EpochLoss(label)
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
    trainer.fit(model, loader)
    train_s = time.time() - t0
    meta = {
        "train_spots": ds.n_spots(),
        "train_seconds": round(train_s, 1),
        "sec_per_epoch": round(train_s / max(args.epochs, 1), 3),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if gpu else 0.0,
        "final_train_loss": cb.rows[-1]["train_loss"] if cb.rows else None,
        "patch_embedding_in_features": in_dim,
        "train_input_check": check,
    }
    return model, trainer, cb.rows, meta


@torch.no_grad()
def predict(model, test_secs, panel, gray: bool):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)
    ds = HER2STSections(test_secs, panel, train=False, gray=gray)
    check = verify_input(ds, expect_gray=gray, panel=panel)
    out = {}
    t0 = time.time()
    for i in range(len(ds)):
        patches, positions, exps, centers = ds[i]
        pred = model(patches.unsqueeze(0).to(dev), positions.unsqueeze(0).to(dev))
        sec = ds.sections[i]
        out[sec] = {
            "pred": pred.squeeze(0).float().cpu().numpy(),
            "truth": exps.numpy(),
            "centers": centers.numpy(),
            "spot_id": np.array(ds.coords[sec]["spot_id"], dtype=object),
            "input_spread": channel_spread_max(patches.numpy()),
        }
    return out, check, round(time.time() - t0, 1)


def write_arm(run_dir: Path, preds, panel, loss_rows, meta: dict) -> None:
    (run_dir / "preds").mkdir(parents=True, exist_ok=True)
    for sec, d in preds.items():
        np.savez_compressed(run_dir / "preds" / f"{sec}.npz",
                            pred=d["pred"], truth=d["truth"],
                            centers=d["centers"], spot_id=d["spot_id"],
                            genes=np.array(panel, dtype=object))
    if loss_rows is not None:
        with open(run_dir / "loss_curve.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "seconds"])
            w.writeheader()
            w.writerows(loss_rows)
    meta["test_input_spread_per_section"] = {s: d["input_spread"] for s, d in preds.items()}
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {run_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="patient", choices=["patient", "section"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)       # 100
    ap.add_argument("--lr", type=float, default=C.LR)             # 1e-5
    ap.add_argument("--n_layers", type=int, default=C.N_LAYERS)   # 8
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--panel", default=str(C.PANEL_FILE))
    ap.add_argument("--arms", default="1,2,3", help="subset, e.g. 3")
    ap.add_argument("--force", action="store_true", help="overwrite finished arms")
    ap.add_argument("--save_ckpt", action="store_true",
                    help="also save state_dicts (~430 MB each; not needed)")
    args = ap.parse_args()

    checksum = assert_canonical()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    panel = her2st.load_panel(args.panel)
    fold = her2st.folds(args.cv)[args.fold]
    fold_dir = f"fold{args.fold:02d}_{fold['name']}"
    root = C.OUT_DIR / args.tag
    want = {int(a) for a in args.arms.split(",")}

    def rd(arm):
        return root / ARMS[arm]["dir"] / fold_dir

    def done(arm):
        return (rd(arm) / "run.json").exists() and not args.force

    print(f"== {args.tag} | cv={args.cv} fold={args.fold} ({fold['name']}) ==")
    print(f"   train sections {len(fold['train'])}  test {fold['test']}")
    print(f"   genes {len(panel)}  epochs {args.epochs}  lr {args.lr}  "
          f"seed {C.SEED}  gray checksum {checksum}", flush=True)

    base = {
        "model": "HisToGene", "tag": args.tag, "cv": args.cv, "fold": args.fold,
        "held_out": fold["name"],
        "train_sections": fold["train"], "test_sections": fold["test"],
        "n_genes": len(panel), "panel": str(args.panel),
        "epochs": args.epochs, "lr": args.lr, "n_layers": args.n_layers,
        "dim": C.DIM, "dropout": C.DROPOUT, "patch_size": C.PATCH_SIZE,
        "n_pos": C.N_POS, "seed": C.SEED,
        "target": "log10(CP10K+1) over panel columns; missing genes zero-filled",
        "scored_at": "last epoch (no validation-based selection)",
    }

    # ---- colour model -> arm1 + arm2 (always written as a pair) ----------
    need12 = bool(want & {1, 2}) and not (done(1) and done(2))
    if need12:
        print("-- colour model (arm1 + arm2)", flush=True)
        model, trainer, rows, tmeta = train_model(fold["train"], panel, False, args, "colour")
        if args.save_ckpt:
            rd(1).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), rd(1) / "state_dict.pt")

        p1, c1, s1 = predict(model, fold["test"], panel, gray=False)
        write_arm(rd(1), p1, panel, rows, {
            **base, "arm": ARMS[1]["dir"], "gray_train": False, "gray_test": False,
            **gray_provenance("none"), **tmeta,
            "test_input_check": c1, "predict_seconds": s1})

        p2, c2, s2 = predict(model, fold["test"], panel, gray=True)
        diff = min(float(np.abs(p1[s]["pred"] - p2[s]["pred"]).max()) for s in p1)
        if diff == 0.0:
            raise RuntimeError("arm2 predictions identical to arm1: gray did not reach the model")
        write_arm(rd(2), p2, panel, None, {
            **base, "arm": ARMS[2]["dir"], "gray_train": False, "gray_test": True,
            **gray_provenance("query"),
            "weights": "same in-process colour model as arm1_colour (not retrained)",
            "train_input_check": tmeta["train_input_check"],
            "test_input_check": c2, "predict_seconds": s2,
            "min_over_sections_max_abs_pred_diff_vs_arm1": diff})
        print(f"   arm2 vs arm1: smallest per-section max|pred diff| = {diff:.4g}  (must be > 0)", flush=True)
        del model, trainer
        torch.cuda.empty_cache()
    elif want & {1, 2}:
        print("-- arm1/arm2 already done, skipping (use --force to redo)")

    # ---- gray model -> arm3 ----------------------------------------------
    if 3 in want and not done(3):
        print("-- grayscale model (arm3)", flush=True)
        model, trainer, rows, tmeta = train_model(fold["train"], panel, True, args, "gray")
        if args.save_ckpt:
            rd(3).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), rd(3) / "state_dict.pt")
        p3, c3, s3 = predict(model, fold["test"], panel, gray=True)
        write_arm(rd(3), p3, panel, rows, {
            **base, "arm": ARMS[3]["dir"], "gray_train": True, "gray_test": True,
            **gray_provenance("all"), **tmeta,
            "test_input_check": c3, "predict_seconds": s3})
        del model, trainer
        torch.cuda.empty_cache()
    elif 3 in want:
        print("-- arm3 already done, skipping (use --force to redo)")

    print(f"-- fold {args.fold} ({fold['name']}) finished", flush=True)


if __name__ == "__main__":
    main()
