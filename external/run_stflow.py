#!/usr/bin/env python
"""Stage 5 driver -- STFlow, depth-normalised run (stflow_lopo_833_normtarget_e20).

Everything about the model is rebuilt from that fold's own run.json ("args"), so the
architecture, normalize_method (panel_cp10k_log1p), prior (gaussian_fitted), n_sample_steps
and eval_seed are exactly the trained run's.  Weights = <fold>/last.pth.

Per fold P:
  prior     per-gene Gaussian refit on the fold's training sections' (normalised) labels --
            same deterministic code as train.py
  her2st    P's sections from the port's UNI cache, predicted in ONE call in the stored
            order and seed (predict() seeds once, then draws section by section) ->
            round-trip vs <fold>/preds/<sec>.npz  (stochastic sampler: tolerance 0.002)
  Visium    UNI (frozen, fp32) on 112-um crops of the resampled image
            (round(112 / um_per_px_her2st) px, white pad) resized to 224 -- the port's own
            crop / to_uni_input / embed; coords = pixel_x, pixel_y at her2st scale (what the
            model saw for her2st); one predict() call per section with the run's eval_seed
  truth     log1p(panel CP10K) -- norm_target.build_target(..., "panel_cp10k_log1p")

usage: cd /workspace/st_benching && python external/run_stflow.py --folds B
needs: pip install -r stflow_port/requirements_stflow.txt   (timm, einops, torch_geometric, ...)
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

sys.path.insert(0, str(K.ST_BENCH / "stflow_port"))
import build_features as BF  # noqa: E402
import config as SC  # noqa: E402
import train as T  # noqa: E402
from dataset import load_cache, split_sections  # noqa: E402
from norm_target import build_target, invert_raw_log1p  # noqa: E402
from stflow_import import build_interpolant, load  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
TAG = "stflow_lopo_833_normtarget_e20"
FEAT_EXT = K.EXT / "features" / "stflow"


def visium_features(sec, um_per_px, device):
    FEAT_EXT.mkdir(parents=True, exist_ok=True)
    f = FEAT_EXT / f"{sec}.npz"
    cnt = K.read_counts(sec, K.VIS_ROOT)
    sp = K.read_spots(sec, K.VIS_ROOT).loc[cnt.index]
    if f.exists():
        z = np.load(f, allow_pickle=True)
        assert list(z["spot_id"]) == list(cnt.index)
        return z["feat"], cnt, sp
    crop_px = int(round(SC.HEST_PATCH_UM / um_per_px))
    d = K.VIS_ROOT / "ST-imgs" / sec[0] / sec
    img = np.asarray(Image.open(d / sorted(os.listdir(d))[0]).convert("RGB"))
    arrs = [BF.to_uni_input(BF.crop(img, x, y, crop_px), False)
            for x, y in zip(sp.pixel_x.values, sp.pixel_y.values)]
    del img
    model = BF.load_uni(device)
    feat = BF.embed(model, arrs, device, 64).astype(np.float32)
    del model; torch.cuda.empty_cache()
    np.savez(f, feat=feat, spot_id=np.array(cnt.index), crop_px=crop_px)
    print(f"  {sec}: UNI features {feat.shape} (crop {crop_px}px = 112 um)", flush=True)
    return feat, cnt, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=K.VISIUM_SECTIONS)
    ap.add_argument("--out", default="/workspace/runs/ext_stflow")
    ap.add_argument("--skip_roundtrip", action="store_true")
    ap.add_argument("--rt_tol", type=float, default=2e-3)
    ap.add_argument("--seed_check", type=int, default=3,
                    help="extra held-out predictions with eval_seed+1..+n: the PCC spread across "
                         "prior draws is the noise floor the round-trip difference is judged against")
    a0 = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = K.load_panel()
    ref = json.loads((K.CALIB / "her2st_scale.json").read_text())

    first = json.load(open(os.path.join(SC.run_dir(TAG), "fold01_B", "run.json")))["args"]
    data, genes, man = load_cache(first["cache_tag"])
    assert set(genes) == set(panel), "STFlow cache genes != panel"
    perm = [genes.index(g) for g in panel]                     # model column -> panel order
    norm = first["normalize_method"]
    assert norm == "panel_cp10k_log1p", f"expected the depth-normalised run, got {norm}"
    for s in data:
        data[s]["labels"] = build_target(invert_raw_log1p(data[s]["labels"]), norm)

    vis = {}
    for sec in a0.sections:
        feat, cnt, sp = visium_features(sec, ref["um_per_px_ref"], device)
        raw = cnt.reindex(columns=genes, fill_value=0).to_numpy(np.float64)   # model gene order
        agg, centres = K.aggregate_counts(raw, list(cnt.index), sec)
        vis[sec] = dict(section=sec, patient=sec[0], features=feat,
                        labels=build_target(raw, norm).astype(np.float32),
                        coords=sp[["pixel_x", "pixel_y"]].to_numpy(np.float32),
                        spot_id=np.array(cnt.index), truth_ps=build_target(agg, norm), centres=centres)

    U = load()
    folds = K.lopo_folds()
    if a0.folds != "all":
        folds = [f for f in folds if f["patient"] in a0.folds.split(",")]
    rt = []
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        src = os.path.join(SC.run_dir(TAG), f"fold{k:02d}_{P}")
        a = Namespace(**json.load(open(os.path.join(src, "run.json")))["args"])
        if not torch.cuda.is_available():
            a.device = "cpu"
        train_s, _, test_s = split_sections(sorted(data), P, None)
        ytr = np.concatenate([data[s]["labels"] for s in train_s], 0).astype(np.float64)
        mu = ytr.mean(0).astype(np.float32)
        sd = np.maximum(ytr.std(0), a.prior_sd_floor).astype(np.float32)
        interp = build_interpolant(a, mu, sd)
        model = U["Denoiser"](a).to(device)
        model.load_state_dict(torch.load(os.path.join(src, "last.pth"), map_location=device,
                                         weights_only=True), strict=True)
        model.eval()
        trainmean = mu[perm]
        out_dir = os.path.join(a0.out, f"fold0{k}_{P}")
        extra = dict(ckpt=os.path.join(src, "last.pth"), eval_seed=a.eval_seed,
                     n_sample_steps=a.n_sample_steps, prior="gaussian_fitted (fold train labels)",
                     target=norm, train=train_s, test=test_s, visium=list(a0.sections))

        if not a0.skip_roundtrip:
            secs = [data[s] for s in test_s]
            preds = T.predict(model, interp, secs, genes, a, a.eval_seed)
            alt = [T.predict(model, interp, secs, genes, a, a.eval_seed + j + 1) for j in range(a0.seed_check)]
            for i_s, (d, pred) in enumerate(zip(secs, preds)):
                s = d["section"]
                K.write_preds(out_dir, s, pred=pred[:, perm], truth=d["labels"][:, perm],
                              spot_ids=list(d["spot_id"]), genes=panel, trainmean=trainmean,
                              model="stflow", fold=k, extra=extra)
                st = np.load(os.path.join(src, "preds", f"{s}.npz"), allow_pickle=True)
                assert [str(x) for x in st["spot_id"]] == [str(x) for x in d["spot_id"]]
                seed_pcc = [float(np.nanmean(K.per_gene_pcc(p[i_s], d["labels"]))) for p in alt]
                rt.append(K.roundtrip_row("stflow", P, s, pred, d["labels"], st["pred"], st["truth"],
                                          seed_pcc_sd=round(float(np.std(seed_pcc + [float(np.nanmean(K.per_gene_pcc(pred, d["labels"])))], ddof=1)), 4) if seed_pcc else np.nan,
                                          seed_pcc_range=round(float(np.ptp(seed_pcc + [float(np.nanmean(K.per_gene_pcc(pred, d["labels"])))])), 4) if seed_pcc else np.nan))
                print("  roundtrip", rt[-1], flush=True)

        for sec, d in vis.items():
            pred = T.predict(model, interp, [d], genes, a, a.eval_seed)[0]
            K.write_preds(out_dir, sec, pred=pred[:, perm], truth=d["labels"][:, perm],
                          spot_ids=list(d["spot_id"]), genes=panel, trainmean=trainmean,
                          truth_ps=d["truth_ps"][:, perm], centre_ids=d["centres"],
                          model="stflow", fold=k, extra=extra)
        print(f"fold {P}: done in {time.time()-t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
    K.record_roundtrip("stflow", rt, a0.rt_tol)
    if rt and a0.seed_check:
        import pandas as pd
        cur = pd.DataFrame(rt)
        cur["abs_diff"] = (cur.pcc_new - cur.pcc_stored).abs()
        within = (cur.abs_diff <= cur.seed_pcc_range).all()
        print(cur[["fold", "section", "pcc_stored", "pcc_new", "abs_diff", "seed_pcc_sd", "seed_pcc_range"]]
              .to_string(index=False))
        print("SEED CHECK:", "every |stored - new| is within the spread of PCC across prior draws "
              "-> difference is sampling noise, not wiring" if within else
              "some |stored - new| exceed the across-draw spread -> tell Claude", flush=True)


if __name__ == "__main__":
    main()
