#!/usr/bin/env python
"""Stage 5 driver -- ST-Net (densenet121_224/top_833, epoch 25), her2st held-out + Visium.

Main-table run (stnet-her2st/run_stnet_833.sh):
  --model densenet121 --window 224 --gene_transform log --norm --gene_list <panel_train.txt>
  target y = log((1 + c_g) / (n + Z)),  n = len(gene.pkl) (whole gene universe), Z = spot total
  over that universe  (stnet/datasets/spatial.py, norm == "norm")
  input  = 224 px crop centred on round(pixel), ToTensor, Normalize(mean, std) where mean/std
           were ESTIMATED FROM 12 SHUFFLED TRAINING BATCHES at run start and only exist in the
           log -> parsed from output/.../top_833/<P>_gene.log
  output columns = gene.pkl order filtered by the gene list (NOT panel order) -> reordered here

Per fold P:
  her2st  P's sections through ST-Net's own Spatial dataset (openslide on the .tif): exact path,
          compared with the stored <P>_25.npz  (round-trip, Stage 4)
          + the same sections through the direct JPEG crop used for Visium, to measure how much
          the TIFF (vips JPEG Q90) vs JPEG input path moves PCC   (reported as dpcc_jpeg)
  Visium  224 px crops from the resampled JPEG (PIL crop, black outside the image = openslide's
          transparent -> RGB), same normalisation, same target with n and Z over the her2st universe

usage: cd /workspace/st_benching && python external/run_stnet.py --folds B
"""
import argparse
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402

STNET = "/workspace/ST-Net"
RUN = f"{STNET}/output/densenet121_224/top_833"
Image.MAX_IMAGE_PIXELS = None
WIN = 224


def per_gene_pcc(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def import_stnet():
    os.chdir(STNET)                       # stnet.config reads ./stnet.cfg
    sys.path.insert(0, STNET)
    try:
        import stnet  # noqa: F401
        import stnet.datasets  # noqa: F401
        return stnet
    except Exception as e:
        raise SystemExit(f"cannot import stnet ({e}).\n  apt-get install -y openslide-tools && "
                         "pip install openslide-python pyyaml scikit-image")


def norm_stats(P):
    txt = open(f"{RUN}/{P}_gene.log").read()
    m = re.findall(r"Estimating mean \(tensor\(\[([^\]]+)\]\)\) and std \(tensor\(\[([^\]]+)\]\)\)", txt)
    if not m:
        raise SystemExit(f"no 'Estimating mean' line in {P}_gene.log")
    mean, std = ([float(v) for v in s.split(",")] for s in m[-1])
    return mean, std


def load_model(P, n_out, dev, stnet):
    model = torchvision.models.densenet121()
    stnet.utils.nn.set_out_features(model, n_out)
    sd = torch.load(f"{RUN}/{P}_checkpoints/epoch_25.pt", map_location="cpu", weights_only=False)["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}   # saved under DataParallel
    model.load_state_dict(sd, strict=True)
    return model.eval().to(dev)


@torch.no_grad()
def forward(model, patches_u8, mean, std, dev, bs=128):
    """patches_u8: (n, 224, 224, 3) uint8 -> ToTensor + Normalize, exactly as torchvision does."""
    m = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(patches_u8), bs):
        x = torch.from_numpy(np.ascontiguousarray(patches_u8[i:i + bs])).to(dev)
        x = x.permute(0, 3, 1, 2).float().div(255.0)
        out.append(model((x - m) / s).float().cpu().numpy())
    return np.concatenate(out)


def crop_all(img_path, px, py):
    img = Image.open(img_path).convert("RGB")
    out = np.zeros((len(px), WIN, WIN, 3), np.uint8)
    for i, (x, y) in enumerate(zip(px, py)):
        out[i] = np.asarray(img.crop((x - WIN // 2, y - WIN // 2, x + WIN // 2, y + WIN // 2)))
    return out


def stnet_target(counts_universe):
    """ST-Net --norm --gene_transform log on counts over the whole gene universe (rows = spots)."""
    n = counts_universe.shape[1]
    Z = counts_universe.sum(1, keepdims=True)
    return np.log((1.0 + counts_universe) / (n + Z))


def visium_universe_counts(sec, universe):
    z = np.load(K.CALIB / f"counts_{sec}.npz", allow_pickle=True)
    X = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    names = [str(g) for g in z["genes"]]
    col = {g: i for i, g in enumerate(universe)}
    rows, cols = [], []
    for j, g in enumerate(names):
        if g in col:
            rows.append(j); cols.append(col[g])
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(names), len(universe)))
    U = np.asarray((X @ M).todense(), dtype=np.float64)        # duplicate symbols summed
    return U, [str(s) for s in z["spot_id"]], len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=K.VISIUM_SECTIONS)
    ap.add_argument("--gene_list", default="/workspace/panels/panel_train.txt")
    ap.add_argument("--out", default="/workspace/runs/ext_stnet")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip_roundtrip", action="store_true")
    ap.add_argument("--rt_tol", type=float, default=1e-3,
                    help="|dPCC| per section; normalisation stats are only logged to 4 decimals")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    stnet = import_stnet()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = K.load_panel()

    # ---- gene universe and column order ------------------------------------------------
    root = stnet.config.SPATIAL_PROCESSED_ROOT
    universe_ensg = pickle.load(open(os.path.join(root, "gene.pkl"), "rb"))
    universe = [str(stnet.utils.ensembl.symbol[g]) for g in universe_ensg]
    glist = open(a.gene_list).read().split()
    out_genes = [g for g, e in zip(universe, universe_ensg) if g in glist or e in glist]
    assert len(out_genes) == len(panel) and set(out_genes) == set(panel), \
        f"model outputs {len(out_genes)} genes; panel {len(panel)}; diff {sorted(set(out_genes) ^ set(panel))[:10]}"
    perm = [out_genes.index(g) for g in panel]            # model column -> panel order
    upos = [universe.index(g) for g in panel]              # universe column of each panel gene
    print(f"gene universe {len(universe)}; model outputs {len(out_genes)} genes (reordered to panel)")

    # ---- Visium inputs and truth, model-independent: build once ------------------------
    vis = {}
    for sec in a.sections:
        sp = K.read_spots(sec, K.VIS_ROOT)
        U, sid, n_mapped = visium_universe_counts(sec, universe)
        sp = sp.loc[sid]
        px = np.round(sp["pixel_x"].values).astype(int)
        py = np.round(sp["pixel_y"].values).astype(int)
        img = K.VIS_ROOT / "ST-imgs" / sec[0] / sec
        patches = crop_all(img / sorted(os.listdir(img))[0], px, py)
        truth = stnet_target(U)[:, upos]
        agg, centres = K.aggregate_counts(U, sid, sec)
        truth_ps = stnet_target(agg)[:, upos]
        vis[sec] = (patches, truth, truth_ps, sid, centres)
        print(f"{sec}: {len(sid)} spots, {n_mapped} Visium features mapped into the her2st universe")

    stored = {P: np.load(f"{RUN}/{P}_25.npz", allow_pickle=True) for P in K.HER2ST_PATIENTS}
    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    rt_rows = []
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        mean, std = norm_stats(P)
        model = load_model(P, len(out_genes), dev, stnet)
        trainmean = np.concatenate([stored[Q]["counts"] for Q in K.HER2ST_PATIENTS if Q != P])[:, perm].mean(0)
        out_dir = os.path.join(out_root, f"fold0{k}_{P}")
        extra = dict(ckpt=f"{RUN}/{P}_checkpoints/epoch_25.pt", norm_mean=mean, norm_std=std,
                     target="log((1+c)/(n+Z)), n=|gene.pkl|, Z over the her2st universe",
                     train=fd["train"], test=fd["test"], visium=list(a.sections))

        if not a.skip_roundtrip:
            tf = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(mean=mean, std=std)])
            ds = stnet.datasets.Spatial([P], tf, window=WIN, gene_filter=glist, norm="norm",
                                        gene_transform="log")
            dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=a.workers)
            P_, Y_, C_, S_, PX_ = [], [], [], [], []
            with torch.no_grad():
                for X, _, y, coord, _, _, sec_, pix, *_ in dl:
                    P_.append(model(X.to(dev)).float().cpu().numpy()); Y_.append(y.numpy())
                    C_.append(coord.numpy()); S_ += list(sec_); PX_.append(pix.numpy())
            pred_all, truth_all = np.concatenate(P_)[:, perm], np.concatenate(Y_)[:, perm]
            coord_all, pix_all, S_ = np.concatenate(C_), np.concatenate(PX_), np.array(S_)
            st = stored[P]
            skey = {(str(s), int(c[0]), int(c[1])): i for i, (s, c) in enumerate(zip(st["section"], st["coord"]))}
            for s in sorted(set(S_)):
                m = S_ == s
                sec = f"{P}{s}"
                sid = [f"{c[0]}x{c[1]}" for c in coord_all[m]]
                pred, truth = pred_all[m], truth_all[m]
                K.write_preds(out_dir, sec, pred=pred, truth=truth, spot_ids=sid, genes=panel,
                              trainmean=trainmean, model="stnet", fold=k, extra=extra)
                sel = [skey[(s, int(c[0]), int(c[1]))] for c in coord_all[m]]
                sp_, st_ = st["predictions"][sel][:, perm], st["counts"][sel][:, perm]
                # same sections through the Visium (direct JPEG) input path
                jpg = os.path.join(stnet.config.SPATIAL_RAW_ROOT, ds.subtype[P], P, f"{P}_{s}.jpg")
                pj = forward(model, crop_all(jpg, pix_all[m][:, 0], pix_all[m][:, 1]), mean, std, dev)[:, perm]
                pcc_new = float(np.nanmean(per_gene_pcc(pred, truth)))
                rt_rows.append(dict(model="stnet", fold=P, section=sec,
                                    max_abs_pred_diff=float(np.abs(pred - sp_).max()),
                                    max_abs_truth_diff=float(np.abs(truth - st_).max()),
                                    pcc_stored=round(float(np.nanmean(per_gene_pcc(sp_, st_))), 4),
                                    pcc_new=round(pcc_new, 4),
                                    dpcc_jpeg=round(float(np.nanmean(per_gene_pcc(pj, truth))) - pcc_new, 4)))
                print("  roundtrip", rt_rows[-1])

        for sec, (patches, truth, truth_ps, sid, centres) in vis.items():
            pred = forward(model, patches, mean, std, dev)[:, perm]
            K.write_preds(out_dir, sec, pred=pred, truth=truth, spot_ids=sid, genes=panel,
                          trainmean=trainmean, truth_ps=truth_ps, centre_ids=centres,
                          model="stnet", fold=k, extra=extra)
        print(f"fold {P}: mean {mean} std {std}  done in {time.time()-t0:.0f}s")
        del model
        torch.cuda.empty_cache()

    if rt_rows:
        p = K.EXT / "roundtrip.tsv"
        cur = pd.DataFrame(rt_rows)
        df = cur
        if p.exists():
            old = pd.read_csv(p, sep="\t")
            df = pd.concat([old[~((old.model == "stnet") & old.fold.isin(cur.fold))], cur])
        df.to_csv(p, sep="\t", index=False)
        bad = cur[(cur.pcc_new - cur.pcc_stored).abs() > a.rt_tol]
        print("ROUNDTRIP", "FAIL" if len(bad) else "PASS", f"({len(bad)} sections off by >{a.rt_tol:g})",
              f"| JPEG-path dPCC range [{cur.dpcc_jpeg.min():+.4f}, {cur.dpcc_jpeg.max():+.4f}]")


if __name__ == "__main__":
    main()
