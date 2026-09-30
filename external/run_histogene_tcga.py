#!/usr/bin/env python
"""TCGA stage B -- HisToGene (histogene_lopo_833_ckpt weights) on TCGA-BRCA whole-slide windows.

Same model, crops and output space as run_histogene_he.py:
  model      vis_model.HisToGene, main-table config (histogene/config.py), runs/histogene_lopo_833_ckpt/fold0k_P/last.ckpt
  input      112 px crops at floor(pixel) of the her2st-scale window JPEG, zero pad (histogene.cache._crop),
             transposed to the repo's (x, y, c) order, raw 0-255 float, flattened -- one forward pass per block
  positions  tcga_common.her2st_blocks: the window's array coords translated to start at (2, 2); windows
             larger than her2st's trained range (x 2-32, y 2-34) are split into blocks that fit.
             Without this, the first row/column (coord 1) and anything beyond 32/34 would index untrained
             position-embedding rows.  (--he-check feeds He's own positions unchanged, to reproduce He.)
  output     raw = log10(CP10K + 1) over the panel (the port's target space)
             lin = max(10**raw - 1, 0)
  no truth   HisToGene reads only patches + positions; ST-cnts are never opened.

All 8 fold models stay on the GPU; sections are the outer loop.  Resumable per section.

--he-check SEC   one He section through THIS crop/forward code with He positions vs the stored He predictions
                 (/workspace/runs/he_histogene/fold0<k>_<P>/preds/<SEC>.npz): PASS = max |diff| < 1e-3.

writes: <out>/fold0<k>_<P>/agg/<SEC>.npz  (+ spots/<SEC>.npz for --spots-for patients)
usage:  cd /workspace/st_benching && python external/run_histogene_tcga.py [--limit 6] [--he-check BC23287_C1]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import run_histogene as R  # noqa: E402  (load_model; imports the histogene port)
import tcga_common as T  # noqa: E402
from run_histogene import C, cache, her2st  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def crops(img_path, pixel_x, pixel_y):
    """run_histogene_he.he_item's patch path: floor(pixel), cache._crop, transpose, flatten (uint8)."""
    img = np.asarray(Image.open(img_path).convert("RGB"))
    px = np.floor(np.asarray(pixel_x)).astype(int)
    py = np.floor(np.asarray(pixel_y)).astype(int)
    r = C.PATCH_R
    patches = np.zeros((len(px), 2 * r, 2 * r, 3), np.uint8)
    for i in range(len(px)):
        patches[i], _ = cache._crop(img, px[i], py[i], r)
    del img
    return np.ascontiguousarray(patches.transpose(0, 2, 1, 3)).reshape(len(px), -1)


@torch.no_grad()
def forward(model, patches_u8, positions, dev):
    assert positions.min() >= 0 and positions.max() < C.N_POS
    x = torch.from_numpy(patches_u8).float().unsqueeze(0).to(dev)
    pos = torch.from_numpy(np.asarray(positions, np.int64)).unsqueeze(0).to(dev)
    return model(x, pos).squeeze(0).float().cpu().numpy()


def he_check(sec, models, dev, he_root):
    import he_common as H
    sp = K.read_spots(sec, H.HE_DATA)
    d = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
    worst = 0.0
    for P, m in models.items():
        z = np.load(os.path.join(he_root, f"fold0{m['fold']}_{P}", "preds", f"{sec}.npz"), allow_pickle=True)
        s = sp.loc[[str(v) for v in z["spot_id"]]]
        patches = crops(d / sorted(os.listdir(d))[0], s.pixel_x.values, s.pixel_y.values)
        pred = forward(m["model"], patches, np.stack([s.x.values, s.y.values], 1), dev)
        diff = float(np.abs(pred - z["pred"]).max())
        worst = max(worst, diff)
        print(f"  he-check {sec} fold {P}: {len(s)} spots, max |diff| {diff:.2e}")
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
    ap.add_argument("--run", default="/workspace/runs/histogene_lopo_833_ckpt")
    ap.add_argument("--out", default="/workspace/runs/tcga_histogene")
    ap.add_argument("--he-check", default=None)
    ap.add_argument("--he-root", default="/workspace/runs/he_histogene")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    torch.backends.cuda.matmul.allow_tf32 = True       # same numerics as the He / Visium drivers
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = her2st.load_panel()
    assert panel == K.load_panel(), "histogene panel != external panel"

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    models = {}
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        ckpt = os.path.join(a.run, f"fold{k:02d}_{P}", "last.ckpt")
        models[P] = dict(fold=k, model=R.load_model(ckpt, len(panel), dev),
                         extra=dict(ckpt=ckpt, patch=C.PATCH_SIZE, train=fd["train"], cohort="tcga",
                                    positions="tcga_common.her2st_blocks (translate to 2,2; split to fit x 2-32, y 2-34)",
                                    raw="log10(CP10K+1) over the panel", inverse="max(10**raw - 1, 0)"))
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
        sds = []
        for P, m in models.items():
            raw = np.zeros((len(sp), len(panel)), np.float32)
            for idx, pos in blocks:
                raw[idx] = forward(m["model"], patches[idx], pos, dev)
            lin = np.maximum(np.power(10.0, raw.astype(np.float64)) - 1.0, 0.0)
            T.write_agg(dirs[P], row, pred_raw=raw, pred_lin=lin, spot_ids=list(sp.index), genes=panel,
                        model="histogene", fold=m["fold"], inverse="max(10**raw - 1, 0)",
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
