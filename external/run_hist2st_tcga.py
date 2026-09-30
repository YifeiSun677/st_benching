#!/usr/bin/env python
"""TCGA stage B -- Hist2ST (hist2st_lopo_833) on TCGA-BRCA whole-slide windows.

Same model, crops, graph and output space as run_hist2st_he.py; model rebuilt from each fold's run.json
"config", weights <fold>/model.pt:
  input      112 px crop at floor(pixel) of the her2st-scale window JPEG, top-left anchored, zero pad,
             permute(0, 3, 2, 1); /255 if the run used scale255 -- one forward pass per block
  positions  tcga_common.her2st_blocks: array coords translated to start at (2, 2); windows larger than
             her2st's trained range (x 2-32, y 2-34) are split into blocks that fit (learned x / y
             embeddings exist only there).  --he-check feeds He's own positions unchanged.
  graph      calcADJ on the block's positions with the run's k and Grid pruning (translation-invariant, so
             identical to calcADJ on the window coords within a block); spots with no neighbour inside the
             pruning radius get a self-loop, exactly as run_hist2st_he.py (TCGA has 1-spot windows)
  output     raw = log10(x / lib * median(lib) + 1)  (the port's target space; the per-section median scale
             multiplies every gene of a spot equally and cancels in the panel-CPM pseudo-bulk)
             lin = max(10**raw - 1, 0)
  no truth   ST-cnts are never opened.

All 8 fold models stay on the GPU; sections are the outer loop.  Resumable per section.

--he-check SEC   one He section through THIS crop/graph/forward code with He positions vs the stored He
                 predictions (/workspace/runs/he_hist2st/fold0<k>_<P>/preds/<SEC>.npz): PASS = max |diff| < 1e-3.

writes: <out>/fold0<k>_<P>/agg/<SEC>.npz  (+ spots/<SEC>.npz for --spots-for patients)
usage:  cd /workspace/st_benching && python external/run_hist2st_tcga.py [--limit 6] [--he-check BC23287_C1]
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
import run_hist2st as R  # noqa: E402  (build_model, forward; imports the hist2st port)
import tcga_common as T  # noqa: E402
from run_hist2st import HC, calcADJ, load_panel  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def crops(img_path, pixel_x, pixel_y):
    """run_hist2st_he.he_item's patch path: floor(pixel), top-left anchored zero pad (uint8 [n, P, P, 3])."""
    im = np.asarray(Image.open(img_path).convert("RGB"))
    Hh, Ww, _ = im.shape
    Rr, Pp = HC.R, HC.PATCH
    px = np.floor(np.asarray(pixel_x)).astype(int)
    py = np.floor(np.asarray(pixel_y)).astype(int)
    patches = np.zeros((len(px), Pp, Pp, 3), np.uint8)
    for i, (x, y) in enumerate(zip(px, py)):
        x0, y0, x1, y1 = max(0, x - Rr), max(0, y - Rr), min(Ww, x + Rr), min(Hh, y + Rr)
        crop = im[y0:y1, x0:x1]
        patches[i, :crop.shape[0], :crop.shape[1]] = crop
    del im
    return patches


def graph(pos, cfg):
    """calcADJ with the run's k and pruning.  Upstream calcADJ walks neighbours 1..k of each spot and
    raises IndexError when a block has <= k spots (He sections never did; TCGA has 1-spot windows):
    k is capped at n - 1, which gives the same graph full k would (every other spot is a candidate,
    Grid pruning unchanged).  n = 1 -> no neighbours -> the self-loop below."""
    pos = np.asarray(pos, np.int64)
    k = min(cfg["neighbor"], len(pos) - 1)
    adj = calcADJ(pos, k, pruneTag=cfg["prune"]).float() if k > 0 else torch.zeros((len(pos), len(pos)))
    iso = np.where(adj.sum(1).numpy() == 0)[0]
    if len(iso):
        adj[iso, iso] = 1.0                    # run_hist2st_he.py's fix: 0/0 NaN otherwise
    return adj, len(iso)


def predict(model, patches_u8, pos, cfg, dev):
    assert np.asarray(pos).min() >= 0 and np.asarray(pos).max() < 64
    patch = torch.from_numpy(patches_u8).permute(0, 3, 2, 1).float()
    if cfg.get("scale255"):
        patch = patch / 255.0
    adj, n_iso = graph(pos, cfg)
    pred = R.forward(model, patch, torch.from_numpy(np.asarray(pos, np.int64)), adj, dev)
    return pred, n_iso


def he_check(sec, models, dev, he_root):
    import he_common as H
    sp = K.read_spots(sec, H.HE_DATA)
    d = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
    worst = 0.0
    for P, m in models.items():
        z = np.load(os.path.join(he_root, f"fold0{m['fold']}_{P}", "preds", f"{sec}.npz"), allow_pickle=True)
        s = sp.loc[[str(v) for v in z["spot_id"]]]
        patches = crops(d / sorted(os.listdir(d))[0], s.pixel_x.values, s.pixel_y.values)
        pred, n_iso = predict(m["model"], patches, s[["x", "y"]].to_numpy(np.int64), m["cfg"], dev)
        diff = float(np.abs(pred - z["pred"]).max())
        worst = max(worst, diff)
        print(f"  he-check {sec} fold {P}: {len(s)} spots, {n_iso} isolated, max |diff| {diff:.2e}")
    print(f"HE-CHECK {'PASS' if worst < 1e-3 else 'FAIL'} (worst {worst:.2e}, tolerance 1e-3)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--batch", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spots-for", nargs="*", default=None)
    ap.add_argument("--run", default="/workspace/runs/hist2st_lopo_833")
    ap.add_argument("--out", default="/workspace/runs/tcga_hist2st")
    ap.add_argument("--he-check", default=None)
    ap.add_argument("--he-root", default="/workspace/runs/he_hist2st")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    panel = load_panel()
    assert panel == K.load_panel()
    print(f"HIST2ST_NORM={HC.NORM}", flush=True)

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    models = {}
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        rdir = os.path.join(a.run, f"fold{k:02d}_{P}")
        cfg = json.load(open(os.path.join(rdir, "run.json")))["config"]
        model = R.build_model(cfg, len(panel), dev)
        model.load_state_dict(torch.load(os.path.join(rdir, "model.pt"), map_location=dev, weights_only=False),
                              strict=True)
        model.eval()
        models[P] = dict(fold=k, model=model, cfg=cfg,
                         extra=dict(ckpt=os.path.join(rdir, "model.pt"), config=cfg, norm=HC.NORM, cohort="tcga",
                                    positions="tcga_common.her2st_blocks (translate to 2,2; split to fit x 2-32, y 2-34)",
                                    graph="calcADJ on block positions, run's k and Grid prune; self-loop on isolated spots",
                                    raw="log10(x/lib*median(lib)+1) over the panel", inverse="max(10**raw - 1, 0)",
                                    train=fd["train"]))
    print(f"loaded {len(models)} fold models on {dev}")

    if a.he_check:
        he_check(a.he_check, models, dev, a.he_root)
        return

    w = T.select_sections(a.sections, a.patients, a.kinds, a.batch, a.limit)
    spot_pats = set(T.default_spot_patients() if a.spots_for is None else a.spots_for)
    print(f"{len(w)} sections, {int(w.n_spots.sum())} spots; full spot matrices kept for {sorted(spot_pats)}")
    t_all, n_all = time.time(), 0
    for row in w.itertuples():
        dirs = {P: os.path.join(out_root, f"fold0{m['fold']}_{P}") for P, m in models.items()}
        if all(T.agg_path(d, row.section).exists() for d in dirs.values()):
            continue
        t0 = time.time()
        sp = T.read_spots(row.section)
        patches = crops(T.image_path(row.section), sp.pixel_x.values, sp.pixel_y.values)
        blocks = T.her2st_blocks(sp.x.values, sp.y.values)
        sds, iso = [], 0
        for P, m in models.items():
            raw = np.zeros((len(sp), len(panel)), np.float32)
            iso = 0
            for idx, pos in blocks:
                raw[idx], n_iso = predict(m["model"], patches[idx], pos, m["cfg"], dev)
                iso += n_iso
            lin = np.maximum(np.power(10.0, raw.astype(np.float64)) - 1.0, 0.0)
            T.write_agg(dirs[P], row, pred_raw=raw, pred_lin=lin, spot_ids=list(sp.index), genes=panel,
                        model="hist2st", fold=m["fold"], inverse="max(10**raw - 1, 0)",
                        save_spots=row.patient in spot_pats, extra=m["extra"])
            sds.append(float(np.median(raw.std(0))) if len(sp) > 1 else float("nan"))
        n_all += len(sp)
        print(f"{row.section}: {len(sp)} spots in {len(blocks)} block(s), {iso} isolated, median across-spot SD "
              f"of raw pred {np.nanmean(sds):.4f}, {time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t_all
    if n_all:
        print(f"DONE {n_all} spots x {len(models)} folds in {dt:.0f}s -> {1000 * dt / n_all:.2f} s per 1000 spots")


if __name__ == "__main__":
    main()
