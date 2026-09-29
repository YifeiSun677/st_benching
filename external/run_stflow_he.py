#!/usr/bin/env python
"""He Stage 4 -- STFlow, depth-normalised run (stflow_lopo_833_normtarget_e20) on He et al. 2020.

Same model, features, prior and target as run_stflow.py (the Visium driver); everything about the
model is rebuilt from each fold's own run.json ("args"), weights <fold>/last.pth.
  prior     per-gene Gaussian refit on the fold's training sections' (normalised) labels
  features  UNI (frozen, fp32) on 112-um crops of the her2st-scale He JPEG
            (round(112 / um_per_px_her2st) px, white pad) resized to 224 -- the port's own
            crop / to_uni_input / embed.  Cached under ext/he/features/stflow/<SEC>.npz
  coords    pixel_x, pixel_y at her2st scale (what the model saw for her2st)
  sampling  one predict() call per section with the run's eval_seed (as the Visium arm)
  truth     log1p(panel CP10K) -- norm_target.build_target(..., "panel_cp10k_log1p")

The her2st held-out sections for the paired comparison come from run_stflow.py (runbook H.1).

writes: <out>/fold0<k>_<P>/preds/<SEC>.npz   (he_common.write_preds, cohort 'he')
usage:  cd /workspace/st_benching && python external/run_stflow_he.py [--folds A,B] [--sections ...]
"""
import argparse
import json
import os
import sys
import time
from argparse import Namespace

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import he_common as H  # noqa: E402
import run_stflow as R  # noqa: E402  (puts the stflow port on sys.path; TAG)
from run_stflow import BF, SC, T, build_interpolant, build_target, invert_raw_log1p, load, load_cache, split_sections  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
FEAT_HE = H.HE / "features" / "stflow"


def he_features(secs, um_per_px, device):
    FEAT_HE.mkdir(parents=True, exist_ok=True)
    crop_px = int(round(SC.HEST_PATCH_UM / um_per_px))
    uni, out = None, {}
    for sec in secs:
        f = FEAT_HE / f"{sec}.npz"
        cnt = K.read_counts(sec, H.HE_DATA)
        sp = K.read_spots(sec, H.HE_DATA).loc[cnt.index]
        if not f.exists():
            t0 = time.time()
            uni = uni or BF.load_uni(device)
            d = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
            img = np.asarray(Image.open(d / sorted(os.listdir(d))[0]).convert("RGB"))
            arrs = [BF.to_uni_input(BF.crop(img, x, y, crop_px), False)
                    for x, y in zip(sp.pixel_x.values, sp.pixel_y.values)]
            del img
            feat = BF.embed(uni, arrs, device, 64).astype(np.float32)
            np.savez(f, feat=feat, spot_id=np.array(cnt.index), crop_px=crop_px)
            print(f"  {sec}: UNI features {feat.shape} (crop {crop_px}px = 112 um), {time.time() - t0:.0f}s",
                  flush=True)
        z = np.load(f, allow_pickle=True)
        assert list(z["spot_id"]) == list(cnt.index), f"{sec}: cached feature rows != counts"
        out[sec] = (z["feat"], cnt, sp)
    if uni is not None:
        del uni
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=None, help="default: every exported He section")
    ap.add_argument("--subtypes", nargs="*", default=None)
    ap.add_argument("--out", default="/workspace/runs/he_stflow")
    ap.add_argument("--features_only", action="store_true")
    a0 = ap.parse_args()
    out_root = os.path.abspath(a0.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = K.load_panel()
    ref = json.loads((K.CALIB / "her2st_scale.json").read_text())

    first = json.load(open(os.path.join(SC.run_dir(R.TAG), "fold01_B", "run.json")))["args"]
    data, genes, man = load_cache(first["cache_tag"])
    assert set(genes) == set(panel), "STFlow cache genes != panel"
    perm = [genes.index(g) for g in panel]                     # model column -> panel order
    norm = first["normalize_method"]
    assert norm == "panel_cp10k_log1p", f"expected the depth-normalised run, got {norm}"
    for s in data:
        data[s]["labels"] = build_target(invert_raw_log1p(data[s]["labels"]), norm)

    exported = {f.name[len("counts_"):-len(".npz")] for f in H.HE_CALIB.glob("counts_*.npz")}
    secs = a0.sections or [s for s in H.he_sections(subtypes=a0.subtypes) if s in exported]
    missing = [s for s in secs if s not in exported]
    if missing:
        raise SystemExit(f"not exported yet: {missing} -- run export_he_like.py")
    feats = he_features(secs, ref["um_per_px_ref"], device)
    if a0.features_only:
        print(f"features cached for {len(feats)} sections under {FEAT_HE}")
        return
    he = {}
    for sec, (feat, cnt, sp) in feats.items():
        raw = cnt.reindex(columns=genes, fill_value=0).to_numpy(np.float64)   # model gene order
        he[sec] = dict(section=sec, patient=H.metadata().loc[sec, "patient"], features=feat,
                       labels=build_target(raw, norm).astype(np.float32),
                       coords=sp[["pixel_x", "pixel_y"]].to_numpy(np.float32), spot_id=np.array(cnt.index))

    U = load()
    folds = K.lopo_folds()
    if a0.folds != "all":
        folds = [f for f in folds if f["patient"] in a0.folds.split(",")]
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        src = os.path.join(SC.run_dir(R.TAG), f"fold{k:02d}_{P}")
        a = Namespace(**json.load(open(os.path.join(src, "run.json")))["args"])
        if not torch.cuda.is_available():
            a.device = "cpu"
        train_s, _, _ = split_sections(sorted(data), P, None)
        ytr = np.concatenate([data[s]["labels"] for s in train_s], 0).astype(np.float64)
        mu = ytr.mean(0).astype(np.float32)
        sd = np.maximum(ytr.std(0), a.prior_sd_floor).astype(np.float32)
        interp = build_interpolant(a, mu, sd)
        model = U["Denoiser"](a).to(device)
        model.load_state_dict(torch.load(os.path.join(src, "last.pth"), map_location=device,
                                         weights_only=True), strict=True)
        model.eval()
        extra = dict(ckpt=os.path.join(src, "last.pth"), eval_seed=a.eval_seed, cohort="he",
                     n_sample_steps=a.n_sample_steps, prior="gaussian_fitted (fold train labels)",
                     target=norm, train=train_s)
        out_dir = os.path.join(out_root, f"fold0{k}_{P}")
        pcc = []
        for sec, d in he.items():
            pred = T.predict(model, interp, [d], genes, a, a.eval_seed)[0]
            H.write_preds(out_dir, sec, pred=pred[:, perm], truth=d["labels"][:, perm],
                          spot_ids=list(d["spot_id"]), genes=panel, trainmean=mu[perm],
                          model="stflow", fold=k, extra=extra)
            pcc.append(np.nanmean(K.per_gene_pcc(pred, d["labels"])))
        print(f"fold {P}: {len(he)} He sections, mean PCC {np.mean(pcc):.4f} "
              f"[{np.min(pcc):.4f}, {np.max(pcc):.4f}], {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
