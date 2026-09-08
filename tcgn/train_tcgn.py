"""
Train TCGN on one fold and save held-out predictions.

Deviations from upstream train.py (all deliberate, benchmark-consistent):
  * fixed epoch budget scored at the LAST epoch; NO best-epoch selection on the
    held-out section (upstream keeps the epoch with best test-section PCC).
  * eval loader uses drop_last=False so EVERY held-out spot is predicted
    (upstream's test loader drops the last partial batch, losing spots).
  * target = 833-panel log10(CP10K+1) via her2_data (no scprep).
Everything else (optimiser Adam lr 1e-5, MSE, batch 32, CMT-Tiny strict=False,
RandomRotation+flip train aug, seed 6) is upstream.

Writes, per fold, into OUT_DIR/<tag>/<fold>/:
  preds/<section>.npz   pred[N,833], truth[N,833], centers[N,2], spot_id[N], genes[833]
  run.json              config, sections, timing, per-epoch train/val MSE
Optionally saves periodic prediction snapshots (--save_every) for the probe.
"""
import os
import io
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
import her2_data as H
from dataset import TCGNDataset
from tcgn_import import load_tcgn


def _predict(model, loader, device):
    model.eval()
    preds, mse_sum, nb = [], 0.0, 0
    lf = nn.MSELoss()
    with torch.no_grad():
        for imgs, genes in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            mse_sum += lf(out, genes.to(device)).item(); nb += 1
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, 0), (mse_sum / max(nb, 1))


def _save_preds(out_dir, test_ds, pred, panel):
    pdir = os.path.join(out_dir, "preds")
    os.makedirs(pdir, exist_ok=True)
    genes = np.asarray(panel)
    for name in np.unique(test_ds.section):
        m = test_ds.section == name
        # centers for this section, in target order
        _, centers, sids = H.section_targets(name, panel)
        np.savez(os.path.join(pdir, name + ".npz"),
                 pred=pred[m].astype(np.float32),
                 truth=test_ds.exps.numpy()[m].astype(np.float32),
                 centers=centers.astype(np.int32),
                 spot_id=test_ds.spot_id[m].astype(str),
                 genes=genes)


def train_fold(fold_name, test_sections, train_sections, tag,
               epochs=C.EPOCHS, save_every=0, device="cuda"):
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    panel = H.load_panel()
    out_dir = os.path.join(C.OUT_DIR, tag, fold_name)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    train_ds = TCGNDataset(train_sections, train=True)
    test_ds = TCGNDataset(test_sections, train=False)
    load_s = time.time() - t0
    print("[%s] train spots=%d  test spots=%d  (load %.1fs)"
          % (fold_name, len(train_ds), len(test_ds), load_s))

    train_loader = DataLoader(train_ds, batch_size=C.BATCH, shuffle=True,
                              num_workers=C.NUM_WORKERS, drop_last=True, pin_memory=True)
    # drop_last=False on eval: keep every held-out spot.
    test_loader = DataLoader(test_ds, batch_size=C.BATCH, shuffle=False,
                             num_workers=C.NUM_WORKERS, drop_last=False, pin_memory=True)

    model = load_tcgn(num_classes=833, load_cmt=True, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=C.WEIGHT_DECAY, amsgrad=False)
    lf = nn.MSELoss()

    curve = []
    ep_times = []
    for ep in range(1, epochs + 1):
        model.train()
        te = time.time(); tr_sum, nb = 0.0, 0
        for imgs, genes in train_loader:
            opt.zero_grad()
            out = model(imgs.to(device))
            loss = lf(out, genes.to(device))
            loss.backward(); opt.step()
            tr_sum += loss.item(); nb += 1
        ep_times.append(time.time() - te)
        if ep == epochs or (save_every and ep % save_every == 0):
            pred, val_mse = _predict(model, test_loader, device)
            curve.append({"epoch": ep, "train_mse": tr_sum / max(nb, 1), "val_mse": val_mse})
            print("  epoch %d/%d  train_mse=%.4f  val_mse=%.4f  (%.1fs/ep)"
                  % (ep, epochs, tr_sum / max(nb, 1), val_mse, ep_times[-1]))
            if save_every and ep % save_every == 0 and ep != epochs:
                snap = os.path.join(out_dir, "snap_ep%d" % ep)
                _save_preds(snap, test_ds, pred, panel)
        else:
            curve.append({"epoch": ep, "train_mse": tr_sum / max(nb, 1), "val_mse": None})

    # final (last-epoch) predictions = the scored ones
    pred, val_mse = _predict(model, test_loader, device)
    _save_preds(out_dir, test_ds, pred, panel)

    peak = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
    run = {
        "model": "TCGN", "tag": tag, "fold": fold_name,
        "test_sections": list(test_sections), "train_sections": list(train_sections),
        "epochs": epochs, "lr": C.LR, "batch": C.BATCH,
        "target_rescale": C.TARGET_RESCALE, "target_log_base": C.TARGET_LOG_BASE,
        "n_train": len(train_ds), "n_test": len(test_ds),
        "sec_per_epoch": float(np.mean(ep_times)), "load_s": load_s,
        "peak_gpu_gb": peak, "final_val_mse": val_mse, "curve": curve,
    }
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2)
    print("[%s] done. mean %.2fs/epoch, peak %.1f GB -> %s"
          % (fold_name, np.mean(ep_times), peak, out_dir))
    return run


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["lopo", "loso"], default="lopo")
    ap.add_argument("--fold", type=int, default=0, help="fold index within the protocol")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--save_every", type=int, default=0, help="probe: snapshot preds every N epochs")
    args = ap.parse_args()

    folds = H.lopo_folds() if args.protocol == "lopo" else H.loso_folds()
    name, te, tr = folds[args.fold]
    tag = args.tag or ("tcgn_%s_833_e%d" % (args.protocol, args.epochs))
    train_fold(name, te, tr, tag, epochs=args.epochs, save_every=args.save_every)
