#!/usr/bin/env python
"""TCGA stage B -- STFlow, depth-normalised run (stflow_lopo_833_normtarget_e20) on TCGA-BRCA windows.

Same model, features, prior and sampling as run_stflow_he.py; everything about the model is rebuilt from
each fold's own run.json ("args"), weights <fold>/last.pth:
  prior     per-gene Gaussian refit on the fold's training sections' (normalised) labels
  features  UNI (frozen) on 112-um crops of the her2st-scale window JPEG (round(112 / um_per_px_her2st) px,
            white pad) resized to 224 -- the port's own crop / to_uni_input / embed.  Not cached
            (450k x 1024 x 4 B = 1.8 GB); each section is embedded once and shared by the 8 folds
  coords    pixel_x, pixel_y at her2st scale (upstream centres them per section, so only scale and extent
            matter).  STFlow samples all spots of a section jointly, so each window is cut with
            tcga_common.her2st_blocks into her2st-sized blocks (<= 31 x 33 spots); one predict() call per
            block with the run's eval_seed, as one call per section for He / Visium
  labels    predict() never feeds labels to the model -- upstream test() only returns them so the port can
            check the row order.  TCGA passes a row-index placeholder (row i = i) that keeps that check
            meaningful; --placeholder-check proves the predictions do not depend on it.
  output    raw = log1p(CP10K over the panel) ('panel_cp10k_log1p'), columns reordered to the panel
            lin = max(expm1(raw), 0)

--he-check SEC        one He section through THIS path (fresh UNI features, placeholder labels, He coords, whole
                      section) vs the stored He predictions (/workspace/runs/he_stflow/...):
                      PASS = max |diff| < 1e-3 (flow sampling is seeded; GPU drift shows up as a small diff)
--placeholder-check   first section of the selection with two different placeholders: must be bit-identical

writes: <out>/fold0<k>_<P>/agg/<SEC>.npz  (+ spots/<SEC>.npz for --spots-for patients)
usage:  cd /workspace/st_benching && python external/run_stflow_tcga.py [--limit 6] [--he-check BC23287_C1]
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
import run_stflow as R  # noqa: E402  (puts the stflow port on sys.path; TAG)
import tcga_common as T  # noqa: E402
from run_stflow import BF, SC, build_interpolant, build_target, invert_raw_log1p, load, load_cache, split_sections  # noqa: E402
from run_stflow import T as TR  # noqa: E402  (the port's train module: predict)

Image.MAX_IMAGE_PIXELS = None


def uni_features(uni, img_path, pixel_x, pixel_y, crop_px, device):
    img = np.asarray(Image.open(img_path).convert("RGB"))
    arrs = [BF.to_uni_input(BF.crop(img, x, y, crop_px), False) for x, y in zip(pixel_x, pixel_y)]
    del img
    return BF.embed(uni, arrs, device, 64).astype(np.float32)


def placeholder(n, n_genes, offset=0.0):
    """Row i = i + offset: distinct rows, so predict()'s row-order check still means something."""
    return np.repeat((np.arange(n, dtype=np.float32) + offset)[:, None], n_genes, axis=1)


def sample(m, feats, coords, name, patient, genes, labels=None):
    if len(feats) == 1:
        # upstream test() computes per-section metrics that need >= 2 spots (scipy: "x and y must have length
        # at least 2").  A 1-spot block is fed as two identical copies and the first row kept.  Unlike TRIPLEX
        # this is an approximation (the two copies draw different prior noise); it touches 2 of 450,442 spots.
        two = sample(m, np.concatenate([feats, feats]), np.concatenate([coords, coords]), name, patient, genes)
        return two[:1]
    d = dict(section=name, patient=patient, features=feats, coords=np.asarray(coords, np.float32),
             labels=placeholder(len(feats), len(genes)) if labels is None else labels,
             spot_id=np.arange(len(feats)).astype(str))
    return TR.predict(m["model"], m["interp"], [d], genes, m["args"], m["args"].eval_seed)[0]


def load_models(folds, device):
    first = json.load(open(os.path.join(SC.run_dir(R.TAG), "fold01_B", "run.json")))["args"]
    data, genes, _ = load_cache(first["cache_tag"])
    norm = first["normalize_method"]
    assert norm == "panel_cp10k_log1p", f"expected the depth-normalised run, got {norm}"
    for s in data:
        data[s]["labels"] = build_target(invert_raw_log1p(data[s]["labels"]), norm)
    U = load()
    models = {}
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        src = os.path.join(SC.run_dir(R.TAG), f"fold{k:02d}_{P}")
        a = Namespace(**json.load(open(os.path.join(src, "run.json")))["args"])
        if not torch.cuda.is_available():
            a.device = "cpu"
        train_s, _, _ = split_sections(sorted(data), P, None)
        ytr = np.concatenate([data[s]["labels"] for s in train_s], 0).astype(np.float64)
        mu = ytr.mean(0).astype(np.float32)
        sd = np.maximum(ytr.std(0), a.prior_sd_floor).astype(np.float32)
        model = U["Denoiser"](a).to(device)
        model.load_state_dict(torch.load(os.path.join(src, "last.pth"), map_location=device, weights_only=True),
                              strict=True)
        model.eval()
        models[P] = dict(fold=k, model=model, interp=build_interpolant(a, mu, sd), args=a,
                         extra=dict(ckpt=os.path.join(src, "last.pth"), eval_seed=a.eval_seed, cohort="tcga",
                                    n_sample_steps=a.n_sample_steps, prior="gaussian_fitted (fold train labels)",
                                    target=norm, train=train_s, raw="log1p(CP10K over the panel)",
                                    inverse="max(expm1(raw), 0)",
                                    blocks="tcga_common.her2st_blocks: one predict() per block"))
    return models, genes


def he_check(sec, uni, models, genes, perm, crop_px, device, he_root):
    import he_common as H
    sp = K.read_spots(sec, H.HE_DATA)
    d = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
    worst, feat, sid_prev = 0.0, None, None
    for P, m in models.items():
        z = np.load(os.path.join(he_root, f"fold0{m['fold']}_{P}", "preds", f"{sec}.npz"), allow_pickle=True)
        sid = [str(v) for v in z["spot_id"]]
        s = sp.loc[sid]
        if sid != sid_prev:
            feat = uni_features(uni, d / sorted(os.listdir(d))[0], s.pixel_x.values, s.pixel_y.values, crop_px, device)
            sid_prev = sid
        pred = sample(m, feat, s[["pixel_x", "pixel_y"]].to_numpy(np.float32), sec, "he", genes)[:, perm]
        diff = float(np.abs(pred - z["pred"]).max())
        worst = max(worst, diff)
        print(f"  he-check {sec} fold {P}: {len(sid)} spots, max |diff| {diff:.2e}, "
              f"corr {np.corrcoef(pred.ravel(), z['pred'].ravel())[0, 1]:.6f}")
    print(f"HE-CHECK {'PASS' if worst < 1e-3 else 'FAIL'} (worst {worst:.2e}, tolerance 1e-3; placeholder labels used)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--batch", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spots-for", nargs="*", default=None)
    ap.add_argument("--out", default="/workspace/runs/tcga_stflow")
    ap.add_argument("--he-check", default=None)
    ap.add_argument("--he-root", default="/workspace/runs/he_stflow")
    ap.add_argument("--placeholder-check", action="store_true")
    a0 = ap.parse_args()
    out_root = os.path.abspath(a0.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = K.load_panel()
    ref = json.loads((K.CALIB / "her2st_scale.json").read_text())
    crop_px = int(round(SC.HEST_PATCH_UM / ref["um_per_px_ref"]))

    folds = K.lopo_folds()
    if a0.folds != "all":
        folds = [f for f in folds if f["patient"] in a0.folds.split(",")]
    models, genes = load_models(folds, device)
    assert set(genes) == set(panel), "STFlow cache genes != panel"
    perm = [genes.index(g) for g in panel]
    uni = BF.load_uni(device)
    print(f"loaded UNI + {len(models)} fold models on {device}; crop {crop_px}px = {SC.HEST_PATCH_UM} um", flush=True)

    if a0.he_check:
        he_check(a0.he_check, uni, models, genes, perm, crop_px, device, os.path.abspath(a0.he_root))
        return

    w = T.select_sections(a0.sections, a0.patients, a0.kinds, a0.batch, a0.limit)
    if a0.placeholder_check:
        row = w.iloc[0]
        sp = T.read_spots(row.section)
        feat = uni_features(uni, T.image_path(row.section), sp.pixel_x.values, sp.pixel_y.values, crop_px, device)
        idx, pos = T.her2st_blocks(sp.x.values, sp.y.values)[0]
        m = models[next(iter(models))]
        c = sp[["pixel_x", "pixel_y"]].to_numpy(np.float32)[idx]
        p1 = sample(m, feat[idx], c, row.section, row.patient, genes)
        p2 = sample(m, feat[idx], c, row.section, row.patient, genes, labels=placeholder(len(idx), len(genes), 1000.0))
        same = np.array_equal(p1, p2)
        print(f"PLACEHOLDER-CHECK {'PASS' if same else 'FAIL'} ({row.section}, {len(idx)} spots, "
              f"max |diff| {float(np.abs(p1 - p2).max()):.2e})")
        return

    spot_pats = set(T.default_spot_patients() if a0.spots_for is None else a0.spots_for)
    print(f"{len(w)} sections, {int(w.n_spots.sum())} spots; full spot matrices kept for {sorted(spot_pats)}")
    t_all, n_all = time.time(), 0
    for row in w.itertuples():
        dirs = {P: os.path.join(out_root, f"fold0{m['fold']}_{P}") for P, m in models.items()}
        if all(T.agg_path(d, row.section).exists() for d in dirs.values()):
            continue
        t0 = time.time()
        sp = T.read_spots(row.section)
        feat = uni_features(uni, T.image_path(row.section), sp.pixel_x.values, sp.pixel_y.values, crop_px, device)
        coords = sp[["pixel_x", "pixel_y"]].to_numpy(np.float32)
        blocks = T.her2st_blocks(sp.x.values, sp.y.values)
        sds = []
        for P, m in models.items():
            raw = np.zeros((len(sp), len(panel)), np.float32)
            for b, (idx, _) in enumerate(blocks):
                raw[idx] = sample(m, feat[idx], coords[idx], f"{row.section}_b{b}", row.patient, genes)[:, perm]
            lin = np.maximum(np.expm1(raw.astype(np.float64)), 0.0)
            T.write_agg(dirs[P], row, pred_raw=raw, pred_lin=lin, spot_ids=list(sp.index), genes=panel,
                        model="stflow", fold=m["fold"], inverse="max(expm1(raw), 0)",
                        save_spots=row.patient in spot_pats, extra=m["extra"])
            sds.append(float(np.median(raw.std(0))) if len(sp) > 1 else float("nan"))
        n_all += len(sp)
        print(f"{row.section}: {len(sp)} spots in {len(blocks)} block(s), median across-spot SD of raw pred "
              f"{np.nanmean(sds):.4f}, {time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t_all
    if n_all:
        print(f"DONE {n_all} spots x {len(models)} folds in {dt:.0f}s -> {1000 * dt / n_all:.2f} s per 1000 spots")


if __name__ == "__main__":
    main()
