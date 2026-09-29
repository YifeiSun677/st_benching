#!/usr/bin/env python
"""He Stage 4 -- Hist2ST (hist2st_lopo_833; --run for the ep1000 sensitivity) on He et al. 2020.

Same model, input and target as run_hist2st.py (the Visium driver); model rebuilt from each
fold's own run.json "config", weights <fold>/model.pt.
  input      one whole section per forward pass, built like build_cache + load_section: 112 px crop
             at floor(pixel) of the her2st-scale He JPEG, top-left anchored, zero pad, same
             permute(0, 3, 2, 1); /255 if the run used scale255
  positions  He array x, y -- legacy-ST 200-um integer units, exactly as her2st (< n_pos 64)
  graph      calcADJ on the SAME array x, y with the run's k and Grid pruning -- identical to
             her2st (He is the same platform; the Visium x_eq workaround is not needed)
  rows       load_section order: spot ids shared by counts and spot file, sorted
  truth      load_section rule: panel counts, lib over the panel, x MEDIAN lib of the section,
             log10(+1)  (HIST2ST_NORM must be the value the run used: median)

The her2st held-out sections for the paired comparison come from run_hist2st.py (runbook F.1).

writes: <out>/fold0<k>_<P>/preds/<SEC>.npz   (he_common.write_preds, cohort 'he')
usage:  cd /workspace/st_benching && python external/run_hist2st_he.py [--folds A,B] [--sections ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import he_common as H  # noqa: E402
import run_hist2st as R  # noqa: E402  (target, build_model, forward; imports the hist2st port)
from run_hist2st import HC, calcADJ, load_panel, load_section  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def he_item(sec, panel, cfg):
    cnt = K.read_counts(sec, H.HE_DATA)
    sp = K.read_spots(sec, H.HE_DATA)
    ids = sorted(set(cnt.index) & set(sp.index))          # load_section: sorted spot ids
    cnt, sp = cnt.loc[ids], sp.loc[ids]
    ori = cnt.reindex(columns=panel).fillna(0.0).to_numpy(np.float64)
    d = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
    im = np.asarray(Image.open(d / sorted(os.listdir(d))[0]).convert("RGB"))
    Hh, Ww, _ = im.shape
    Rr, P = HC.R, HC.PATCH
    patches = np.zeros((len(ids), P, P, 3), np.uint8)
    for i, (px, py) in enumerate(np.floor(sp[["pixel_x", "pixel_y"]].to_numpy()).astype(int)):
        x0, y0, x1, y1 = max(0, px - Rr), max(0, py - Rr), min(Ww, px + Rr), min(Hh, py + Rr)
        crop = im[y0:y1, x0:x1]
        patches[i, :crop.shape[0], :crop.shape[1]] = crop  # build_cache's top-left zero pad
    del im
    arr = sp[["x", "y"]].to_numpy(np.int64)
    if arr.max() >= 64 or arr.min() < 0:
        raise SystemExit(f"{sec}: array position outside [0, 64)")
    adj = calcADJ(arr, cfg["neighbor"], pruneTag=cfg["prune"]).float()
    return dict(patches=patches, pos=torch.from_numpy(arr), adj=adj, truth=R.target(ori),
                ids=ids, degree=float(adj.sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=None, help="default: every exported He section")
    ap.add_argument("--subtypes", nargs="*", default=None)
    ap.add_argument("--run", default="/workspace/runs/hist2st_lopo_833")
    ap.add_argument("--out", default="/workspace/runs/he_hist2st")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = load_panel()
    assert panel == K.load_panel()
    cfg = json.load(open(os.path.join(a.run, "fold01_B", "run.json")))["config"]
    print(f"config: {cfg}\nHIST2ST_NORM={HC.NORM}", flush=True)

    exported = {f.name[len("counts_"):-len(".npz")] for f in H.HE_CALIB.glob("counts_*.npz")}
    secs = a.sections or [s for s in H.he_sections(subtypes=a.subtypes) if s in exported]
    missing = [s for s in secs if s not in exported]
    if missing:
        raise SystemExit(f"not exported yet: {missing} -- run export_he_like.py")
    t0 = time.time()
    items = {s: he_item(s, panel, cfg) for s in secs}      # graph uses the run-wide k / prune
    deg = [v["degree"] for v in items.values()]
    print(f"built {len(items)} He sections in {time.time() - t0:.0f}s; graph mean degree "
          f"{np.mean(deg):.2f} [{np.min(deg):.2f}, {np.max(deg):.2f}]", flush=True)

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        rdir = os.path.join(a.run, f"fold{k:02d}_{P}")
        cfg_k = json.load(open(os.path.join(rdir, "run.json")))["config"]
        assert (cfg_k["neighbor"], cfg_k["prune"]) == (cfg["neighbor"], cfg["prune"]), \
            f"fold {P} graph config differs from fold B -- rebuild graphs per fold"
        model = R.build_model(cfg_k, len(panel), device)
        model.load_state_dict(torch.load(os.path.join(rdir, "model.pt"), map_location=device,
                                         weights_only=False), strict=True)
        model.eval()
        trainmean = np.concatenate([load_section(s, panel)["exp"] for s in fd["train"]]).mean(0)
        extra = dict(ckpt=os.path.join(rdir, "model.pt"), config=cfg_k, norm=HC.NORM, cohort="he",
                     positions="He array x/y", graph="calcADJ on array x/y, same k and Grid prune",
                     train=fd["train"])
        out_dir = os.path.join(out_root, f"fold0{k}_{P}")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        pcc = []
        for sec, v in items.items():
            patch = torch.from_numpy(v["patches"]).permute(0, 3, 2, 1).float()
            if cfg_k.get("scale255"):
                patch = patch / 255.0
            pred = R.forward(model, patch, v["pos"], v["adj"], device)
            H.write_preds(out_dir, sec, pred=pred, truth=v["truth"], spot_ids=v["ids"], genes=panel,
                          trainmean=trainmean, model="hist2st", fold=k, extra=extra)
            pcc.append(np.nanmean(K.per_gene_pcc(pred, v["truth"])))
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        print(f"fold {P}: {len(items)} He sections, mean PCC {np.mean(pcc):.4f} "
              f"[{np.min(pcc):.4f}, {np.max(pcc):.4f}], peak GPU {peak:.1f} GB, {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
