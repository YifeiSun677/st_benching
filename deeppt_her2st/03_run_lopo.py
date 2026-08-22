#!/usr/bin/env python
"""
Step 3 -- leave-one-patient-out: AE + predictor, refit inside every fold.

Aligned to the released DeepPT code (12AE/1main_AE.py, 13DeepPT_train/*).
Hyperparameters below are transplanted, not guessed:
    n_hiddens 512, dropout 0.2, lr 1e-4 (both stages), Adam with NO weight
    decay, batch 32, max_epochs 500, MLP patience 50, seed 42, MSE loss,
    AE trained for a fixed 500 epochs with no early stopping.

TWO DELIBERATE DEVIATIONS, both to be stated in methods:

  1. LEAKAGE. The original fits the AE on a random 90/10 split of the POOLED
     feature set, then compresses everything -- including slides later used
     as test data. That is transductive. Here the AE is refit inside each
     fold on TRAINING-PATIENT SPOTS ONLY, which is stricter and appropriate
     for a benchmark whose subject is honest LOPO generalisation.

  2. MODEL SELECTION. The original's fit() returns the LAST epoch (patience
     50, no weight restoration). Here the best-validation-PCC epoch is also
     recorded. Per-epoch predictions are saved either way, so both can be
     scored without retraining.

NOT a deviation, and important: features are fed to the AE RAW, unstandardised.
The decoder ends in ReLU, so it can only reconstruct non-negative inputs;
z-scoring would floor the reconstruction loss and degrade the code.

Output per fold:
    preds/<P>/<P>_<epoch>.npz   counts [n_spots, 833], spot_id, section
    preds/<P>/history.csv       train/val loss and mean val PCC per epoch
    ckpt/<P>_{ae,mlp}.pt

Usage:
    python 03_run_lopo.py --features .../features_raw --targets .../targets \
        --out /workspace/deeppt/results/deeppt_833_raw --tag raw
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from deeppt_models import AE, Predictor

PATIENTS = list("ABCDEFGH")


def mean_gene_pcc(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean per-gene Pearson r -- the original's `valid_coef`, the quantity
    its early stopping maximises (utils.compute_coef_slope)."""
    a = pred - pred.mean(0, keepdims=True)
    b = true - true.mean(0, keepdims=True)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    r = np.full(pred.shape[1], np.nan)
    ok = den > 1e-12
    r[ok] = (a * b).sum(0)[ok] / den[ok]
    return float(np.nanmean(r))


def load_split(feat_dir, targ_dir):
    secs = sorted(f[:-4] for f in os.listdir(feat_dir)
                  if f.endswith(".npy") and not f.endswith("_patches.npy"))
    X, Y, meta = {}, {}, {}
    for p in PATIENTS:
        xs, ys, ms = [], [], []
        for s in [s for s in secs if s.startswith(p)]:
            xs.append(np.load(os.path.join(feat_dir, f"{s}.npy")))
            ys.append(np.load(os.path.join(targ_dir, f"{s}.npy")))
            sp = pd.read_csv(os.path.join(feat_dir, f"{s}_spots.csv"))
            ms.append(pd.DataFrame({"spot_id": sp["spot_id"].astype(str),
                                    "section": s}))
        if not xs:
            continue
        X[p] = np.concatenate(xs).astype(np.float32)
        Y[p] = np.concatenate(ys).astype(np.float32)
        meta[p] = pd.concat(ms, ignore_index=True)
        assert len(X[p]) == len(Y[p]) == len(meta[p])
    return X, Y, meta


def train_ae(Xtr, Xva, args, device):
    """1main_AE.py: fixed 500 epochs, Adam lr 1e-4, MSE, no early stopping."""
    ae = AE(Xtr.shape[1], args.d_code).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=args.ae_lr)
    lossf = nn.MSELoss()
    tr = torch.from_numpy(Xtr).to(device)
    va = torch.from_numpy(Xva).to(device)

    best, best_sd, bad = np.inf, None, 0
    for ep in range(args.ae_epochs):
        ae.train()
        perm = torch.randperm(len(tr), device=device)
        for i in range(0, len(tr), args.batch):
            b = tr[perm[i:i + args.batch]]
            opt.zero_grad()
            loss = lossf(ae(b)[0], b)
            loss.backward()
            opt.step()
        ae.eval()
        with torch.no_grad():
            vl = lossf(ae(va)[0], va).item()
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_sd = {k: v.clone() for k, v in ae.state_dict().items()}
        else:
            bad += 1
            if args.ae_patience and bad >= args.ae_patience:
                break
    # match the original: keep the FINAL weights unless --ae-restore-best
    if args.ae_restore_best and best_sd is not None:
        ae.load_state_dict(best_sd)
    return ae.eval(), best, ep + 1


def run_fold(P, X, Y, meta, args, device):
    val_p = PATIENTS[(PATIENTS.index(P) + 1) % len(PATIENTS)]
    train_p = [p for p in PATIENTS if p not in (P, val_p)]
    print(f"\n=== fold {P} | val={val_p} | train={''.join(train_p)} ===")

    Xtr = np.concatenate([X[p] for p in train_p])
    Ytr = np.concatenate([Y[p] for p in train_p])
    Xva, Yva = X[val_p], Y[val_p]
    Xte, Yte = X[P], Y[P]
    # NO standardisation -- the ReLU decoder requires non-negative inputs.

    # ---- (iii) AE, fitted on training patients only --------------------
    t0 = time.time()
    ae, ae_val, ae_eps = train_ae(Xtr, Xva, args, device)
    print(f"  AE: {ae_eps} epochs, val MSE {ae_val:.5f}, {time.time()-t0:.0f}s")

    with torch.no_grad():
        enc = lambda a: ae.encode(torch.from_numpy(a).to(device))
        Ztr, Zva, Zte = enc(Xtr), enc(Xva), enc(Xte)

    # ---- (iv) predictor: Linear -> Dropout -> Linear, mean-bias init ----
    mlp = Predictor(args.d_code, args.d_hidden, Ytr.shape[1], args.dropout,
                    bias_init=Ytr.mean(0)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr)
    lossf = nn.MSELoss()
    ytr = torch.from_numpy(Ytr).to(device)
    yva = torch.from_numpy(Yva).to(device)

    pdir = os.path.join(args.out, "preds", P)
    os.makedirs(pdir, exist_ok=True)
    hist, best_pcc, best_ep, bad = [], -1.0, 0, 0

    for ep in range(1, args.mlp_epochs + 1):
        mlp.train()
        perm = torch.randperm(len(Ztr), device=device)
        tl = 0.0
        for i in range(0, len(Ztr), args.batch):
            j = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(mlp(Ztr[j]), ytr[j])
            loss.backward()
            opt.step()
            tl += loss.item() * len(j)
        tl /= len(Ztr)

        mlp.eval()
        with torch.no_grad():
            pv = mlp(Zva)
            vl = lossf(pv, yva).item()
            vpcc = mean_gene_pcc(pv.cpu().numpy(), Yva)
            pred = mlp(Zte).cpu().numpy()

        np.savez_compressed(
            os.path.join(pdir, f"{P}_{ep}.npz"),
            counts=pred.astype(np.float32),
            spot_id=meta[P]["spot_id"].values,
            section=meta[P]["section"].values,
        )
        hist.append({"epoch": ep, "train_mse": tl, "val_mse": vl,
                     "val_gene_pcc": vpcc})

        # original criterion: maximise mean validation per-gene PCC
        if vpcc > best_pcc:
            best_pcc, best_ep, bad = vpcc, ep, 0
            torch.save(mlp.state_dict(),
                       os.path.join(args.out, "ckpt", f"{P}_mlp.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stopping at epoch {ep}")
                break

    pd.DataFrame(hist).to_csv(os.path.join(pdir, "history.csv"), index=False)
    torch.save(ae.state_dict(), os.path.join(args.out, "ckpt", f"{P}_ae.pt"))
    last_ep = hist[-1]["epoch"]
    print(f"  best val PCC {best_pcc:+.4f} @ epoch {best_ep} "
          f"(last epoch {last_ep})")
    return {"patient": P, "val_patient": val_p,
            "best_epoch": best_ep, "last_epoch": last_ep,
            "best_val_gene_pcc": best_pcc, "ae_epochs": ae_eps,
            "n_test_spots": len(Xte)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="raw")
    # --- transplanted from the released code ---
    ap.add_argument("--d-code", type=int, default=512)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--ae-lr", type=float, default=1e-4)
    ap.add_argument("--mlp-lr", type=float, default=1e-4)
    ap.add_argument("--ae-epochs", type=int, default=500)
    ap.add_argument("--mlp-epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    # --- knobs the original does not expose ---
    ap.add_argument("--ae-patience", type=int, default=0,
                    help="0 = off, matching the original's fixed 500 epochs")
    ap.add_argument("--ae-restore-best", action="store_true",
                    help="off by default: the original keeps final AE weights")
    ap.add_argument("--folds", nargs="*", default=PATIENTS)
    args = ap.parse_args()

    # utils.init_random_seed(42)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(vars(args) | {"device": device}, fh, indent=2)
    print(json.dumps(vars(args), indent=2))

    X, Y, meta = load_split(args.features, args.targets)
    print(f"[data] {sum(len(v) for v in X.values())} spots, "
          f"{Y['A'].shape[1]} genes, patients {sorted(X)}")

    rows = [run_fold(P, X, Y, meta, args, device) for P in args.folds]
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "run_summary.csv"),
                              index=False)
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
