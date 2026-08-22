#!/usr/bin/env python
"""
Step 3 — leave-one-patient-out: AE + predictor, refit inside every fold.

LEAKAGE RULE (the single most important line in this file):
    the feature scaler AND the autoencoder are fitted on TRAINING-PATIENT
    SPOTS ONLY. Fitting either on all 13,620 spots leaks held-out tissue into
    the representation, inflates every number, and is invisible downstream.

Fold structure, for held-out patient P:
    test  = P
    val   = the next patient alphabetically (wrapping) -- used for early stopping
    train = the remaining 6

Per-epoch predictions for the held-out patient are saved (as in your ST-Net
run 2), so the scoring epoch can be changed without retraining.

Output per fold:
    preds/<P>/<P>_<epoch>.npz   counts [n_spots, 833], spot_id, section
    preds/<P>/history.csv       train/val loss per epoch
    ckpt/<P>_{scaler,ae,mlp}.pt

Usage:
    python 03_run_lopo.py --features .../features_raw --targets .../targets \
        --out /workspace/deeppt/results/deeppt_833 --tag raw
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


def load_split(feat_dir, targ_dir):
    """All sections -> arrays keyed by patient."""
    secs = sorted(f[:-4] for f in os.listdir(feat_dir)
                  if f.endswith(".npy") and not f.endswith("_patches.npy"))
    X, Y, meta = {}, {}, {}
    for p in PATIENTS:
        xs, ys, ms = [], [], []
        for s in [s for s in secs if s.startswith(p)]:
            xs.append(np.load(os.path.join(feat_dir, f"{s}.npy")))
            ys.append(np.load(os.path.join(targ_dir, f"{s}.npy")))
            sp = pd.read_csv(os.path.join(feat_dir, f"{s}_spots.csv"))
            ms.append(pd.DataFrame({"spot_id": sp["spot_id"].astype(str), "section": s}))
        if not xs:
            continue
        X[p] = np.concatenate(xs).astype(np.float32)
        Y[p] = np.concatenate(ys).astype(np.float32)
        meta[p] = pd.concat(ms, ignore_index=True)
        assert len(X[p]) == len(Y[p]) == len(meta[p])
    return X, Y, meta


def train_ae(Xtr, Xva, args, device):
    ae = AE(Xtr.shape[1], args.d_code).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=args.ae_lr, weight_decay=args.wd)
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
            rec, _ = ae(b)
            loss = lossf(rec, b)
            loss.backward()
            opt.step()
        ae.eval()
        with torch.no_grad():
            vl = lossf(ae(va)[0], va).item()
        if vl < best - 1e-6:
            best, best_sd, bad = vl, {k: v.clone() for k, v in ae.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
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

    # ---- scaler: TRAIN ONLY -------------------------------------------
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr, Xva, Xte = [(a - mu) / sd for a in (Xtr, Xva, Xte)]

    # ---- (iii) AE: TRAIN ONLY -----------------------------------------
    t0 = time.time()
    ae, ae_val, ae_eps = train_ae(Xtr, Xva, args, device)
    print(f"  AE: {ae_eps} epochs, val MSE {ae_val:.5f}, {time.time()-t0:.0f}s")

    with torch.no_grad():
        enc = lambda a: ae.encode(torch.from_numpy(a).to(device))
        Ztr, Zva, Zte = enc(Xtr), enc(Xva), enc(Xte)

    # ---- (iv) predictor -----------------------------------------------
    mlp = Predictor(args.d_code, args.d_hidden, Ytr.shape[1], args.dropout).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr, weight_decay=args.wd)
    lossf = nn.MSELoss()
    ytr = torch.from_numpy(Ytr).to(device)
    yva = torch.from_numpy(Yva).to(device)

    pdir = os.path.join(args.out, "preds", P)
    os.makedirs(pdir, exist_ok=True)
    hist, best, best_ep, bad = [], np.inf, 0, 0

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
            vl = lossf(mlp(Zva), yva).item()
            pred = mlp(Zte).cpu().numpy()

        # save EVERY epoch -- lets you rescore without retraining
        np.savez_compressed(
            os.path.join(pdir, f"{P}_{ep}.npz"),
            counts=pred.astype(np.float32),
            spot_id=meta[P]["spot_id"].values,
            section=meta[P]["section"].values,
        )
        hist.append({"epoch": ep, "train_mse": tl, "val_mse": vl})

        if vl < best - 1e-6:
            best, best_ep, bad = vl, ep, 0
            torch.save(mlp.state_dict(), os.path.join(args.out, "ckpt", f"{P}_mlp.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {ep}")
                break

    pd.DataFrame(hist).to_csv(os.path.join(pdir, "history.csv"), index=False)
    torch.save({"mu": mu, "sd": sd}, os.path.join(args.out, "ckpt", f"{P}_scaler.pt"))
    torch.save(ae.state_dict(), os.path.join(args.out, "ckpt", f"{P}_ae.pt"))
    print(f"  best val MSE {best:.5f} @ epoch {best_ep}")
    return {"patient": P, "val_patient": val_p, "best_epoch": best_ep,
            "best_val_mse": best, "n_test_spots": len(Xte), "ae_epochs": ae_eps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="raw")
    ap.add_argument("--d-code", type=int, default=512)
    ap.add_argument("--d-hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--ae-lr", type=float, default=1e-4)
    ap.add_argument("--mlp-lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--ae-epochs", type=int, default=300)
    ap.add_argument("--mlp-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", nargs="*", default=PATIENTS)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)

    # log the full config, exactly as you did for ST-Net run 1
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(vars(args) | {"device": device}, fh, indent=2)
    print(json.dumps(vars(args), indent=2))

    X, Y, meta = load_split(args.features, args.targets)
    print(f"[data] {sum(len(v) for v in X.values())} spots, "
          f"{Y['A'].shape[1]} genes, patients {sorted(X)}")

    rows = [run_fold(P, X, Y, meta, args, device) for P in args.folds]
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "run_summary.csv"), index=False)
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
