#!/usr/bin/env python
"""Stage 5 driver -- TRIPLEX (triplex_lopo_833_e20_ckpt, final.pt = last of 20 epochs).

Per fold P, the model's three branches are fed exactly as the port feeds her2st:
  target    224 px crop around the spot, white pad (BLEEP's crop, the cache TRIPLEX used),
            ToTensor + ImageNet Normalize
  global    CIGAR ResNet18 512-d feature of every spot's 224 px crop, + position -> APEG
  neighbour CIGAR features of the spots at array offsets (-2..+2, -2..+2) around the target,
            centre = token 12, mask 0 where no spot.  her2st: exact grid lookup (port).
            Visium: the SAME physical offsets (multiples of 200 um): nearest Visium spot to
            (x_eq + dc, y_eq + dr) if within 0.25 units (50 um), else masked
  position  her2st (array_col, array_row); Visium (x_eq, y_eq) -- same 200-um units
  truth     BLEEP/TRIPLEX target: panel counts -> CPM over the panel -> natural log1p

APEG GRID (important): upstream APEG infers its grid size (W, H) from the positions on its
FIRST forward call and caches it on the module (not in the checkpoint).  Stored predictions
used the grid set during training.  This driver infers (W, H) on every training section of
the fold, pins the modal value on all APEG modules, prints the spread, and uses it for
her2st and Visium alike (Visium positions are in the same units, so the same grid applies).
Colliding spots share a grid cell and are averaged by APEG itself (scatter_add / count).

Output columns are reordered from the cache panel order to panel_833.txt order.

usage: cd /workspace/st_benching && python external/run_triplex.py --folds B
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402

sys.path.insert(0, str(K.ST_BENCH))
from triplex import config as TC, her2st as TH  # noqa: E402
from triplex.build_features import _OFFSETS  # noqa: E402
from triplex.cigar import build_cigar_encoder, encode_patches  # noqa: E402
from triplex.dataset import TriTestSections, _TEST_TF, _load_section  # noqa: E402
from triplex.load_ckpt import load_fold  # noqa: E402

sys.path.insert(0, str(K.ST_BENCH / "bleep"))
import her2st_dataset as BD  # noqa: E402  (white-padded 224 crop + CPM->log1p, as the cache)

TAG = "triplex_lopo_833_e20_ckpt"
FEAT_EXT = K.EXT / "features" / "triplex"


def _read_counts_any(root, section):          # the pod's her2st copy is .tsv, Visium .tsv.gz
    import pandas as pd
    for ext in (".tsv.gz", ".tsv"):
        p = os.path.join(root, "ST-cnts", f"{section}{ext}")
        if os.path.exists(p):
            return pd.read_csv(p, sep="\t", index_col=0)
    raise FileNotFoundError(section)


BD.read_counts = _read_counts_any


def apeg_modules(model):
    return [m for m in model.modules() if type(m).__name__ == "APEG"]


def pin_grid(model, positions_by_section):
    mods = apeg_modules(model)
    if not mods:
        return None, {}
    grids = collections.Counter(tuple(mods[0].infer_grid_size(torch.as_tensor(p).float(), rounding_factor=20))
                                for p in positions_by_section)
    grid = grids.most_common(1)[0][0]
    for m in mods:
        m.grid_size = grid
    return grid, dict(grids)


def visium_inputs(sec, panel_cache, device):
    """Patches, CIGAR global features, neighbour tensor, positions, truth -- built once."""
    FEAT_EXT.mkdir(parents=True, exist_ok=True)
    q = BD.Her2stCLIPDataset(str(K.VIS_ROOT), [sec], panel_cache, is_train=False, verbose=False)
    S = q.sections[0]
    sid = [str(s) for s in S.spot_ids]
    patches = np.stack([S.patch(i) for i in range(len(sid))])            # (N,224,224,3) uint8
    f = FEAT_EXT / f"{sec}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        assert [str(s) for s in z["spot_id"]] == sid
        glob_feat = z["feat"]
    else:
        enc = build_cigar_encoder(device=device)
        glob_feat = encode_patches(enc, patches, device=device).astype(np.float32)
        del enc; torch.cuda.empty_cache()
        np.savez(f, feat=glob_feat, spot_id=np.array(sid))
    sp = K.read_spots(sec, K.VIS_ROOT).loc[sid]
    xy = sp[["x_eq", "y_eq"]].to_numpy(np.float64)
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    neigh = np.zeros((len(sid), 25, glob_feat.shape[1]), np.float32)
    mask = np.zeros((len(sid), 25), np.int64)
    for k_, (dr, dc) in enumerate(_OFFSETS):                            # (row, col) offsets
        dist, j = tree.query(xy + np.array([dc, dr]), k=1)
        ok = dist < 0.25
        if dr == 0 and dc == 0:
            assert ok.all() and (j == np.arange(len(sid))).all()
        neigh[ok, k_] = glob_feat[j[ok]]
        mask[ok, k_] = 1
    cnt = BD.read_counts(str(K.VIS_ROOT), sec).loc[sid]
    raw, _ = BD.align_to_panel(cnt, panel_cache)
    agg, centres = K.aggregate_counts(raw.to_numpy(np.float64), sid, sec)
    return dict(patches=patches, glob=glob_feat, neigh=neigh, mask=mask,
                pos=xy.astype(np.float32), truth=q.expression_matrix(),
                truth_ps=BD.cpm_log1p(agg.astype(np.float32)), sid=sid, centres=centres,
                n_neigh=float(mask.sum(1).mean()))


@torch.no_grad()
def run_model(model, patches, neigh, mask, glob, pos, device):
    img = torch.stack([_TEST_TF(p) for p in patches], 0).to(device)
    res = model(img=img, mask=torch.as_tensor(mask).long().to(device),
                neighbor_emb=torch.as_tensor(neigh).float().to(device),
                position=torch.as_tensor(pos).float().to(device),
                global_emb=torch.as_tensor(glob).float().unsqueeze(0).to(device))
    return res["logits"].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=K.VISIUM_SECTIONS)
    ap.add_argument("--out", default="/workspace/runs/ext_triplex")
    ap.add_argument("--skip_roundtrip", action="store_true")
    ap.add_argument("--rt_tol", type=float, default=1e-3)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(str(K.ST_BENCH))                  # upstream init looks for ./weights/cigar (run.sh)
    panel = K.load_panel()
    panel_cache = TH.load_panel()               # expression.npy / model column order
    assert set(panel_cache) == set(panel)
    perm = [panel_cache.index(g) for g in panel]

    vis = {}
    for sec in a.sections:
        vis[sec] = visium_inputs(sec, panel_cache, device)
        print(f"  {sec}: {len(vis[sec]['sid'])} spots, neighbours/spot {vis[sec]['n_neigh']:.1f} "
              f"(her2st interior = 25)", flush=True)

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    rt = []
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        model, meta = load_fold(TAG, P, device=device)
        tr = {s: _load_section(s, panel_cache) for s in fd["train"]}
        grid, spread = pin_grid(model, [tr[s]["coords"] for s in fd["train"]])
        trainmean = np.concatenate([tr[s]["expr"] for s in fd["train"]]).mean(0)[perm]
        out_dir = os.path.join(a.out, f"fold0{k}_{P}")
        extra = dict(ckpt=os.path.join(TC.CKPT_DIR, TAG, f"fold_{P}", "final.pt"),
                     apeg_grid=grid, apeg_grid_by_train_section=str(spread),
                     neighbours="Visium: nearest spot within 50 um of each 200-um offset",
                     train=fd["train"], test=fd["test"], visium=list(a.sections))
        print(f"  [{P}] APEG grid pinned to {grid}; per training section: {spread}", flush=True)

        if not a.skip_roundtrip:
            te = TriTestSections(fd["test"], panel_cache)
            for s in fd["test"]:
                b = te.section_batch(s, device=device)
                with torch.no_grad():
                    pred = model(img=b["img"], mask=b["mask"], neighbor_emb=b["neighbor_emb"],
                                 position=b["position"], global_emb=b["global_emb"])["logits"].float().cpu().numpy()
                sid = [str(x) for x in b["spot_id"]]
                K.write_preds(out_dir, s, pred=pred[:, perm], truth=b["label"][:, perm], spot_ids=sid,
                              genes=panel, trainmean=trainmean, model="triplex", fold=k, extra=extra)
                st = np.load(os.path.join(TC.OUTPUT_DIR, TAG, f"fold_{P}", "preds", f"{s}.npz"),
                             allow_pickle=True)
                assert [str(x) for x in st["spot_id"]] == sid
                rt.append(K.roundtrip_row("triplex", P, s, pred, b["label"], st["pred"], st["truth"]))
                print("  roundtrip", rt[-1], flush=True)
                del b; torch.cuda.empty_cache()

        for sec, v in vis.items():
            pred = run_model(model, v["patches"], v["neigh"], v["mask"], v["glob"], v["pos"], device)
            K.write_preds(out_dir, sec, pred=pred[:, perm], truth=v["truth"][:, perm], spot_ids=v["sid"],
                          genes=panel, trainmean=trainmean, truth_ps=v["truth_ps"][:, perm],
                          centre_ids=v["centres"], model="triplex", fold=k, extra=extra)
            torch.cuda.empty_cache()
        print(f"fold {P}: done in {time.time()-t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
    K.record_roundtrip("triplex", rt, a.rt_tol)


if __name__ == "__main__":
    main()
