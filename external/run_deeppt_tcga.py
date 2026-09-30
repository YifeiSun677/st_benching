#!/usr/bin/env python
"""TCGA stage B -- DeepPT (deeppt_833_raw checkpoints) on TCGA-BRCA whole-slide windows.

Same encoder, weights, epoch rule and output space as run_deeppt_he.py:
  encoder   frozen ResNet50 (DeepPT_original/ResNet50_IMAGENET1K_V2.pt), 224 px crops at round(pixel)
            of the her2st-scale window JPEG, zero pad, ImageNet normalisation, fp16 autocast
            -- the port's own 01_extract_features.build_encoder / encode and her2st_io.crop_patches
  AE / MLP  <P>_ae.pt, <P>_mlp.pt (best-val epoch, default) -- use the same --run / --mlp as He
  output    raw = log10(CP10K + 1), CP10K over all detected genes (the port's 02_build_targets space)
            lin = max(10**raw - 1, 0)  -- DeepPT's own inverse (the MLP can dip slightly below 0)
  no truth  DeepPT reads only the image; TCGA ST-cnts are never opened.

Features are NOT cached (450k spots x 2048 x 4 B = 3.7 GB): each section is encoded once and pushed
straight through the 8 fold heads.  Resumable per section.

--he-check SEC   one He section through THIS code path (fresh crop + encode) vs the stored He
                 predictions (/workspace/runs/he_deeppt/fold0<k>_<P>/preds/<SEC>.npz).  The encoder runs
                 in fp16, so expect ~1e-3, not 0; PASS = max |diff| < 1e-2 and every fold's
                 pred-vs-stored correlation > 0.9999.

writes: <out>/fold0<k>_<P>/agg/<SEC>.npz  (+ spots/<SEC>.npz for --spots-for patients)
usage:  cd /workspace/st_benching && python external/run_deeppt_tcga.py [--limit 6] [--he-check BC23287_C1]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import run_deeppt as R  # noqa: E402  (FX encoder, io.crop_patches, AE, Predictor)
import tcga_common as T  # noqa: E402
from run_deeppt import AE, FX, TARG, WEIGHTS, Predictor, io  # noqa: E402


def load_heads(run, mlp_rule, folds, n_genes, dev):
    heads = {}
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        ae = AE(2048, 512).to(dev)
        ae.load_state_dict(torch.load(os.path.join(run, "ckpt", f"{P}_ae.pt"), map_location=dev))
        mlp = Predictor(512, 512, n_genes, 0.2).to(dev)
        mlp_file = os.path.join(run, "ckpt", f"{P}_mlp.pt" if mlp_rule == "best" else f"{P}_mlp_last.pt")
        mlp.load_state_dict(torch.load(mlp_file, map_location=dev))
        hist = pd.read_csv(os.path.join(run, "preds", P, "history.csv"))
        if mlp_rule == "best":
            epoch = int(hist.epoch[hist.val_gene_pcc.cummax().diff().fillna(1).gt(0)].iloc[-1])
        else:
            epoch = int(hist.epoch.iloc[-1])
        ae.eval(); mlp.eval()
        heads[P] = dict(fold=k, ae=ae, mlp=mlp,
                        extra=dict(ae=os.path.join(run, "ckpt", f"{P}_ae.pt"), mlp=mlp_file, cohort="tcga",
                                   epoch_rule=f"{mlp_rule} (epoch {epoch})", train=fd["train"],
                                   raw="log10(CP10K+1), library over all detected genes",
                                   inverse="max(10**raw - 1, 0)"))
    return heads


def run_head(h, feat, dev):
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(feat, np.float32)).to(dev)
        return h["mlp"](h["ae"].encode(x)).cpu().numpy()


def he_check(sec, net, heads, dev, he_root):
    import he_common as H
    sp = K.read_spots(sec, H.HE_DATA)
    d = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
    img = Image.open(d / sorted(os.listdir(d))[0]).convert("RGB")
    worst, ok = 0.0, True
    feat, sid_prev = None, None
    for P, h in heads.items():
        z = np.load(os.path.join(he_root, f"fold0{h['fold']}_{P}", "preds", f"{sec}.npz"), allow_pickle=True)
        sid = [str(s) for s in z["spot_id"]]
        if sid != sid_prev:
            feat = FX.encode(io.crop_patches(img, sp.loc[sid][["pixel_x", "pixel_y"]]), net, dev, 256).astype(np.float32)
            sid_prev = sid
        pred = run_head(h, feat, dev)
        d_ = float(np.abs(pred - z["pred"]).max())
        c = float(np.corrcoef(pred.ravel(), z["pred"].ravel())[0, 1])
        worst, ok = max(worst, d_), ok and c > 0.9999
        print(f"  he-check {sec} fold {P}: {len(sid)} spots, max |diff| {d_:.2e}, corr {c:.6f}")
    img.close()
    print(f"HE-CHECK {'PASS' if (worst < 1e-2 and ok) else 'FAIL'} (worst {worst:.2e}; tolerance 1e-2, corr > 0.9999)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--batch", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spots-for", nargs="*", default=None)
    ap.add_argument("--run", default="/workspace/deeppt/results/deeppt_833_raw")
    ap.add_argument("--mlp", choices=["best", "last"], default="best")
    ap.add_argument("--out", default="/workspace/runs/tcga_deeppt")
    ap.add_argument("--he-check", default=None)
    ap.add_argument("--he-root", default="/workspace/runs/he_deeppt")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    panel = K.load_panel()
    assert open(os.path.join(TARG, "genes.txt")).read().split() == panel, "targets/genes.txt != panel"

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    heads = load_heads(a.run, a.mlp, folds, len(panel), dev)
    net = FX.build_encoder(WEIGHTS, dev)
    print(f"loaded encoder + {len(heads)} fold heads on {dev}: "
          + ", ".join(f"{P} {h['extra']['epoch_rule']}" for P, h in heads.items()))

    if a.he_check:
        he_check(a.he_check, net, heads, dev, a.he_root)
        return

    w = T.select_sections(a.sections, a.patients, a.kinds, a.batch, a.limit)
    spot_pats = set(T.default_spot_patients() if a.spots_for is None else a.spots_for)
    print(f"{len(w)} sections, {int(w.n_spots.sum())} spots; full spot matrices kept for {sorted(spot_pats)}")
    t_all, n_all = time.time(), 0
    for row in w.itertuples():
        dirs = {P: os.path.join(out_root, f"fold0{h['fold']}_{P}") for P, h in heads.items()}
        if all(T.agg_path(d, row.section).exists() for d in dirs.values()):
            continue
        t0 = time.time()
        sp = T.read_spots(row.section)
        img = Image.open(T.image_path(row.section)).convert("RGB")
        feat = FX.encode(io.crop_patches(img, sp[["pixel_x", "pixel_y"]]), net, dev, 256).astype(np.float32)
        img.close()
        sds = []
        for P, h in heads.items():
            raw = run_head(h, feat, dev)
            lin = np.maximum(np.power(10.0, raw.astype(np.float64)) - 1.0, 0.0)
            T.write_agg(dirs[P], row, pred_raw=raw, pred_lin=lin, spot_ids=list(sp.index), genes=panel,
                        model="deeppt", fold=h["fold"], inverse="max(10**raw - 1, 0)",
                        save_spots=row.patient in spot_pats, extra=h["extra"])
            sds.append(float(np.median(raw.std(0))) if len(sp) > 1 else float("nan"))
        n_all += len(sp)
        print(f"{row.section}: {len(sp)} spots, median across-spot SD of raw pred {np.nanmean(sds):.4f}, "
              f"{time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t_all
    if n_all:
        print(f"DONE {n_all} spots x {len(heads)} folds in {dt:.0f}s -> {1000 * dt / n_all:.2f} s per 1000 spots")


if __name__ == "__main__":
    main()
