#!/usr/bin/env python
"""Stage 5 driver -- HisToGene (histogene_lopo_833_ckpt weights), her2st held-out + Visium.

Per fold P:
  model     = vis_model.HisToGene built with the main-table config (histogene/config.py),
              weights from runs/histogene_lopo_833_ckpt/fold0k_P/last.ckpt
  her2st    = P's sections through the port's OWN cached dataset (HER2STSections) -> round-trip
              against the stored preds of the same run (same weights: expect an exact match)
  Visium    = one whole section per forward pass, built exactly like HER2STSections builds an
              item: 112 px crops (floor(pixel), zero pad), transpose to the repo's (x, y, c)
              order, raw 0-255 float, flattened; positions = x_int / y_int (her2st-equivalent
              200-um units) -- NOT the spot file's x, y, which are Visium array indices
  truth     = the port's own target, histogene.her2st.expression: log10(CP10K + 1) over the
              panel columns, missing genes zero-filled
  trainmean = mean of the cached training-patient expression

usage: cd /workspace/st_benching && python external/run_histogene.py --folds B
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

sys.path.insert(0, str(K.ST_BENCH))
from histogene import cache, config as C, her2st  # noqa: E402
from histogene.dataset import HER2STSections  # noqa: E402

sys.path.insert(0, str(C.HISTOGENE_REPO))
from vis_model import HisToGene  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def per_gene_pcc(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def load_model(ckpt, n_genes, device):
    model = HisToGene(patch_size=C.PATCH_SIZE, n_layers=C.N_LAYERS, n_genes=n_genes, dim=C.DIM,
                      learning_rate=C.LR, dropout=C.DROPOUT, n_pos=C.N_POS)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    model.load_state_dict(sd, strict=True)
    return model.eval().to(device)


def visium_item(sec, panel):
    """Mirror of HER2STSections.__getitem__ for a Visium section in the visium_like root."""
    cnt = K.read_counts(sec, K.VIS_ROOT)
    pos = K.read_spots(sec, K.VIS_ROOT)
    meta = cnt.join(pos)                                  # same join as her2st.read_meta
    meta = meta.dropna(subset=["pixel_x", "pixel_y"])
    truth = her2st.expression(meta, panel)               # port's own transform
    px = np.floor(meta["pixel_x"].values).astype(int)
    py = np.floor(meta["pixel_y"].values).astype(int)
    img = np.asarray(Image.open(her2st_image(sec)).convert("RGB"))
    r = C.PATCH_R
    patches = np.zeros((len(meta), 2 * r, 2 * r, 3), np.uint8)
    for i in range(len(meta)):
        patches[i], _ = cache._crop(img, px[i], py[i], r)
    del img
    x = torch.from_numpy(np.ascontiguousarray(patches.transpose(0, 2, 1, 3))).float().flatten(1)
    positions = np.stack([meta["x_int"].values, meta["y_int"].values], 1).astype(np.int64)
    if positions.max() >= C.N_POS:
        raise SystemExit(f"{sec}: integer position {positions.max()} >= n_pos {C.N_POS}")
    return x, torch.from_numpy(positions), truth, list(meta.index), meta


def her2st_image(sec):
    d = K.VIS_ROOT / "ST-imgs" / sec[0] / sec
    return d / sorted(os.listdir(d))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=K.VISIUM_SECTIONS)
    ap.add_argument("--run", default="/workspace/runs/histogene_lopo_833_ckpt")
    ap.add_argument("--out", default="/workspace/runs/ext_histogene")
    ap.add_argument("--skip_roundtrip", action="store_true")
    ap.add_argument("--rt_tol", type=float, default=1e-4)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True       # same numerics as train.py's predict
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = her2st.load_panel()
    assert panel == K.load_panel(), "histogene panel != external panel"
    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    visium_items = {s: visium_item(s, panel) for s in a.sections}   # model-independent, build once
    rt_rows = []

    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        rdir = os.path.join(a.run, f"fold{k:02d}_{P}")
        ckpt = os.path.join(rdir, "last.ckpt")
        out_dir = os.path.join(a.out, f"fold0{k}_{P}")
        model = load_model(ckpt, len(panel), dev)
        trainmean = np.concatenate([cache.load_expr(panel, s) for s in fd["train"]]).mean(0)
        extra = dict(ckpt=ckpt, positions="x_int/y_int (200-um units)", patch=C.PATCH_SIZE,
                     train=fd["train"], test=fd["test"], visium=list(a.sections))

        with torch.no_grad():
            if not a.skip_roundtrip:
                ds = HER2STSections(fd["test"], panel, train=False)
                for i, sec in enumerate(ds.sections):
                    patches, positions, exps, _ = ds[i]
                    pred = model(patches.unsqueeze(0).to(dev), positions.unsqueeze(0).to(dev)).squeeze(0).float().cpu().numpy()
                    truth = exps.numpy()
                    sid = [str(s) for s in ds.coords[sec]["spot_id"]]
                    K.write_preds(out_dir, sec, pred=pred, truth=truth, spot_ids=sid, genes=panel,
                                  trainmean=trainmean, model="histogene", fold=k, extra=extra)
                    z = np.load(os.path.join(rdir, "preds", f"{sec}.npz"), allow_pickle=True)
                    row = {str(s): j for j, s in enumerate(z["spot_id"])}
                    sel = [row[s] for s in sid]
                    sp, st = z["pred"][sel], z["truth"][sel]
                    rt_rows.append(dict(model="histogene", fold=P, section=sec,
                                        max_abs_pred_diff=float(np.abs(pred - sp).max()),
                                        max_abs_truth_diff=float(np.abs(truth - st).max()),
                                        pcc_stored=round(float(np.nanmean(per_gene_pcc(sp, st))), 4),
                                        pcc_new=round(float(np.nanmean(per_gene_pcc(pred, truth))), 4)))
                    print("  roundtrip", rt_rows[-1])

            for sec, (x, positions, truth, sid, meta) in visium_items.items():
                try:
                    pred = model(x.unsqueeze(0).to(dev), positions.unsqueeze(0).to(dev)).squeeze(0).float().cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    raise SystemExit(f"{sec}: OOM on a whole-section pass ({len(sid)} spots) -- "
                                     "needs the windowed variant; tell Claude")
                raw = meta.reindex(columns=panel, fill_value=0)[panel].values.astype(np.float64)
                agg, centres = K.aggregate_counts(raw, sid, sec)
                truth_ps = her2st.expression(pd.DataFrame(agg, columns=panel), panel)
                K.write_preds(out_dir, sec, pred=pred, truth=truth, spot_ids=sid, genes=panel,
                              trainmean=trainmean, truth_ps=truth_ps, centre_ids=centres,
                              model="histogene", fold=k, extra=extra)
        if dev.type == "cuda":
            print(f"  peak GPU {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        print(f"fold {P}: done in {time.time()-t0:.0f}s")
        del model
        torch.cuda.empty_cache()

    if rt_rows:
        p = K.EXT / "roundtrip.tsv"
        df = pd.DataFrame(rt_rows)
        if p.exists():
            old = pd.read_csv(p, sep="\t")
            df = pd.concat([old[~((old.model == "histogene") & old.fold.isin(df.fold))], df])
        df.to_csv(p, sep="\t", index=False)
        cur = pd.DataFrame(rt_rows)
        bad = cur[(cur.pcc_new - cur.pcc_stored).abs() > a.rt_tol]
        print("ROUNDTRIP", "FAIL" if len(bad) else "PASS", f"({len(bad)} sections off by >{a.rt_tol:g})")


if __name__ == "__main__":
    main()
