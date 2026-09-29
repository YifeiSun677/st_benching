#!/usr/bin/env python
"""He Stage 4 -- HisToGene (histogene_lopo_833_ckpt weights) on He et al. 2020 sections.

Same model, input and target as run_histogene.py (the Visium driver); only the cohort differs:
  model      vis_model.HisToGene with the main-table config (histogene/config.py),
             weights runs/histogene_lopo_833_ckpt/fold0k_P/last.ckpt
  input      one whole section per forward pass, built like HER2STSections builds an item:
             112 px crops (floor(pixel), zero pad) of the her2st-scale He JPEG, transposed to the
             repo's (x, y, c) order, raw 0-255 float, flattened
  positions  He spot file x, y -- already legacy-ST array units (200 um), exactly what her2st
             sections feed the position embedding (Visium needed x_int/y_int instead)
  truth      the port's own target, histogene.her2st.expression: log10(CP10K + 1) over the panel,
             missing genes zero-filled
  trainmean  mean of the cached training-patient expression

Patches for all He sections are cropped once (uint8, ~1.5 GB RAM) and reused by every fold.
The her2st held-out sections for the paired comparison come from run_histogene.py (runbook C.1).

writes: <out>/fold0<k>_<P>/preds/<SEC>.npz   (he_common.write_preds, cohort 'he')
usage:  cd /workspace/st_benching && python external/run_histogene_he.py [--folds A,B] [--sections ...]
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
import he_common as H  # noqa: E402
import run_histogene as R  # noqa: E402  (load_model, per_gene_pcc; imports the histogene port)
from run_histogene import C, cache, her2st  # noqa: E402


def he_item(sec, panel):
    """Mirror of run_histogene.visium_item for a He section; returns uint8 patches."""
    cnt = K.read_counts(sec, H.HE_DATA)
    pos = K.read_spots(sec, H.HE_DATA)
    meta = cnt.join(pos)                                  # same join as her2st.read_meta
    meta = meta.dropna(subset=["pixel_x", "pixel_y"])
    truth = her2st.expression(meta, panel)               # port's own transform
    px = np.floor(meta["pixel_x"].values).astype(int)
    py = np.floor(meta["pixel_y"].values).astype(int)
    d = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
    img = np.asarray(Image.open(d / sorted(os.listdir(d))[0]).convert("RGB"))
    r = C.PATCH_R
    patches = np.zeros((len(meta), 2 * r, 2 * r, 3), np.uint8)
    for i in range(len(meta)):
        patches[i], _ = cache._crop(img, px[i], py[i], r)
    del img
    patches = np.ascontiguousarray(patches.transpose(0, 2, 1, 3)).reshape(len(meta), -1)
    positions = np.stack([meta["x"].values, meta["y"].values], 1).astype(np.int64)
    if positions.max() >= C.N_POS or positions.min() < 0:
        raise SystemExit(f"{sec}: array position outside [0, {C.N_POS})")
    return patches, positions, truth, [str(s) for s in meta.index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=None, help="default: every exported He section")
    ap.add_argument("--subtypes", nargs="*", default=None)
    ap.add_argument("--run", default="/workspace/runs/histogene_lopo_833_ckpt")
    ap.add_argument("--out", default="/workspace/runs/he_histogene")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)

    torch.backends.cuda.matmul.allow_tf32 = True       # same numerics as train.py's predict
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = her2st.load_panel()
    assert panel == K.load_panel(), "histogene panel != external panel"

    exported = {f.name[len("counts_"):-len(".npz")] for f in H.HE_CALIB.glob("counts_*.npz")}
    secs = a.sections or [s for s in H.he_sections(subtypes=a.subtypes) if s in exported]
    missing = [s for s in secs if s not in exported]
    if missing:
        raise SystemExit(f"not exported yet: {missing} -- run export_he_like.py")
    t0 = time.time()
    items = {s: he_item(s, panel) for s in secs}         # model-independent, build once
    print(f"cropped {len(items)} He sections, {sum(len(v[3]) for v in items.values())} spots, "
          f"{time.time() - t0:.0f}s", flush=True)

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        ckpt = os.path.join(a.run, f"fold{k:02d}_{P}", "last.ckpt")
        model = R.load_model(ckpt, len(panel), dev)
        trainmean = np.concatenate([cache.load_expr(panel, s) for s in fd["train"]]).mean(0)
        extra = dict(ckpt=ckpt, positions="He array x, y (200-um units)", patch=C.PATCH_SIZE,
                     train=fd["train"], cohort="he")
        out_dir = os.path.join(out_root, f"fold0{k}_{P}")
        pcc = []
        with torch.no_grad():
            for sec, (patches, positions, truth, sid) in items.items():
                x = torch.from_numpy(patches).float().unsqueeze(0).to(dev)
                pos = torch.from_numpy(positions).unsqueeze(0).to(dev)
                pred = model(x, pos).squeeze(0).float().cpu().numpy()
                H.write_preds(out_dir, sec, pred=pred, truth=truth, spot_ids=sid, genes=panel,
                              trainmean=trainmean, model="histogene", fold=k, extra=extra)
                pcc.append(np.nanmean(R.per_gene_pcc(pred, truth)))
        peak = f", peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB" if dev.type == "cuda" else ""
        print(f"fold {P}: {len(items)} He sections, mean PCC {np.mean(pcc):.4f} "
              f"[{np.min(pcc):.4f}, {np.max(pcc):.4f}]{peak}, {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
