#!/usr/bin/env python
"""Stage 5 driver -- Hist2ST (hist2st_lopo_833, 350 epochs; --run for the ep1000 sensitivity).

Model rebuilt with the fold's own run.json "config"; weights = <fold>/model.pt.

Per fold P:
  her2st    P's sections through the port's own Her2stHist2ST dataset (112 px cache,
            permute(0,3,2,1) axis fix, Grid-pruned 4-NN graph on array coords) ->
            round-trip vs <fold>/preds/<sec>.npz (same weights: expect an exact match)
  Visium    one whole section per forward pass, built like build_cache + load_section:
            112 px crop at floor(pixel), top-left anchored, zero pad; same permute;
            positions for the x/y embeddings = x_int / y_int (200-um units, < n_pos 64);
            graph = calcADJ on x_eq / y_eq (continuous 200-um units) with the SAME k and
            Grid threshold (<= 2.0 units) -- on Visium the 4 nearest neighbours sit ~0.5
            units away, so the graph keeps its degree but spans a smaller physical radius
  truth     the port's load_section rule: panel counts, lib over the panel, x MEDIAN lib of
            the section, log10(+1)  (HIST2ST_NORM must be the value the run used: median)
  OOM on a whole Visium section -> the script stops and says so (then a windowed variant)

usage: cd /workspace/st_benching && python external/run_hist2st.py --folds B
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

sys.path.insert(0, str(K.ST_BENCH))
from hist2st import config as HC  # noqa: E402
from hist2st.dataset import Her2stHist2ST  # noqa: E402
from hist2st.her2st import load_panel, load_section  # noqa: E402

sys.path.insert(0, str(HC.HIST2ST_REPO))
from graph_construction import calcADJ  # noqa: E402
from HIST2ST import Hist2ST  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def target(ori):
    """load_section's exp: lib over the panel, x median lib, log10(+1)."""
    ori = np.asarray(ori, np.float64)
    lib = ori.sum(1)
    lib[lib == 0] = 1.0
    scale = np.median(lib) if HC.NORM == "median" else 1e4
    return np.log10(ori / lib[:, None] * scale + 1.0).astype(np.float32)


def visium_item(sec, panel, cfg):
    cnt = K.read_counts(sec, K.VIS_ROOT)
    sp = K.read_spots(sec, K.VIS_ROOT)
    ids = sorted(set(cnt.index) & set(sp.index))          # load_section: sorted spot ids
    cnt, sp = cnt.loc[ids], sp.loc[ids]
    ori = cnt.reindex(columns=panel).fillna(0.0).to_numpy(np.float64)
    d = K.VIS_ROOT / "ST-imgs" / sec[0] / sec
    im = np.asarray(Image.open(d / sorted(os.listdir(d))[0]).convert("RGB"))
    Hh, Ww, _ = im.shape
    R, P = HC.R, HC.PATCH
    patches = np.zeros((len(ids), P, P, 3), np.uint8)
    for i, (px, py) in enumerate(np.floor(sp[["pixel_x", "pixel_y"]].to_numpy()).astype(int)):
        x0, y0, x1, y1 = max(0, px - R), max(0, py - R), min(Ww, px + R), min(Hh, py + R)
        crop = im[y0:y1, x0:x1]
        patches[i, :crop.shape[0], :crop.shape[1]] = crop  # build_cache's top-left zero pad
    del im
    patch = torch.from_numpy(patches).permute(0, 3, 2, 1).float()
    if cfg.get("scale255"):
        patch = patch / 255.0
    pos = sp[["x_int", "y_int"]].to_numpy(np.int64)
    assert pos.max() < 64, f"{sec}: x_int/y_int >= n_pos"
    adj = calcADJ(sp[["x_eq", "y_eq"]].to_numpy(np.float64), cfg["neighbor"], pruneTag=cfg["prune"]).float()
    agg, centres = K.aggregate_counts(ori, ids, sec)
    return dict(patch=patch, pos=torch.from_numpy(pos), adj=adj, truth=target(ori),
                truth_ps=target(agg), ids=ids, centres=centres,
                degree=float(adj.sum(1).mean()))


def build_model(cfg, n_genes, device):
    m = Hist2ST(depth1=cfg["depth1"], depth2=cfg["depth2"], depth3=cfg["depth3"],
                n_genes=n_genes, learning_rate=cfg["lr"], label=None,
                kernel_size=cfg["kernel"], patch_size=cfg["patch"], fig_size=HC.PATCH,
                heads=cfg["heads"], channel=cfg["channel"], dropout=cfg["dropout"],
                zinb=cfg["zinb"], nb=False, bake=cfg["bake"], lamb=cfg["lamb"],
                policy=cfg["policy"]).to(device)
    m.log = lambda *a, **k: None
    m.log_dict = lambda *a, **k: None
    return m


@torch.no_grad()
def forward(model, patch, pos, adj, device):
    o = model(patch.unsqueeze(0).to(device), pos.unsqueeze(0).to(device), adj.to(device))
    pred = o[0] if isinstance(o, (tuple, list)) else o
    pred = pred.squeeze(0) if pred.dim() == 3 else pred
    return pred.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=K.VISIUM_SECTIONS)
    ap.add_argument("--run", default="/workspace/runs/hist2st_lopo_833")
    ap.add_argument("--out", default="/workspace/runs/ext_hist2st")
    ap.add_argument("--skip_roundtrip", action="store_true")
    ap.add_argument("--rt_tol", type=float, default=1e-4)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = load_panel()
    assert panel == K.load_panel()
    cfg = json.load(open(os.path.join(a.run, "fold01_B", "run.json")))["config"]
    print(f"config: {cfg}\nHIST2ST_NORM={HC.NORM}", flush=True)

    vis = {}
    for sec in a.sections:
        vis[sec] = visium_item(sec, panel, cfg)
        print(f"  {sec}: {len(vis[sec]['ids'])} spots, graph mean degree {vis[sec]['degree']:.2f}", flush=True)

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    rt = []
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        rdir = os.path.join(a.run, f"fold{k:02d}_{P}")
        cfg_k = json.load(open(os.path.join(rdir, "run.json")))["config"]
        model = build_model(cfg_k, len(panel), device)
        model.load_state_dict(torch.load(os.path.join(rdir, "model.pt"), map_location=device,
                                         weights_only=False), strict=True)
        model.eval()
        trainmean = np.concatenate([load_section(s, panel)["exp"] for s in fd["train"]]).mean(0)
        out_dir = os.path.join(a.out, f"fold0{k}_{P}")
        extra = dict(ckpt=os.path.join(rdir, "model.pt"), config=cfg_k, norm=HC.NORM,
                     positions="x_int/y_int", graph="calcADJ on x_eq/y_eq, same k and Grid<=2",
                     train=fd["train"], test=fd["test"], visium=list(a.sections))
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

        if not a.skip_roundtrip:
            ds = Her2stHist2ST(fd["test"], neighs=cfg_k["neighbor"], prune=cfg_k["prune"],
                               scale255=cfg_k.get("scale255", False))
            for i, s in enumerate(fd["test"]):
                patch, pos, exp, adj, ori, sf, ctr = ds[i]
                pred = forward(model, patch, pos, adj, device)
                sid = [str(x) for x in ds.data[s]["spot_id"]]
                K.write_preds(out_dir, s, pred=pred, truth=exp.numpy(), spot_ids=sid, genes=panel,
                              trainmean=trainmean, model="hist2st", fold=k, extra=extra)
                st = np.load(os.path.join(rdir, "preds", f"{s}.npz"), allow_pickle=True)
                assert [str(x) for x in st["spot_id"]] == sid
                rt.append(K.roundtrip_row("hist2st", P, s, pred, exp.numpy(), st["pred"], st["truth"]))
                print("  roundtrip", rt[-1], flush=True)

        for sec, v in vis.items():
            try:
                pred = forward(model, v["patch"], v["pos"], v["adj"], device)
            except torch.cuda.OutOfMemoryError:
                raise SystemExit(f"{sec}: OOM on a whole-section pass ({len(v['ids'])} spots) -- "
                                 "needs the windowed variant; tell Claude")
            K.write_preds(out_dir, sec, pred=pred, truth=v["truth"], spot_ids=v["ids"], genes=panel,
                          trainmean=trainmean, truth_ps=v["truth_ps"], centre_ids=v["centres"],
                          model="hist2st", fold=k, extra=extra)
        peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        print(f"fold {P}: done in {time.time()-t0:.0f}s, peak GPU {peak:.1f} GB", flush=True)
        del model; torch.cuda.empty_cache()
    K.record_roundtrip("hist2st", rt, a.rt_tol)


if __name__ == "__main__":
    main()
