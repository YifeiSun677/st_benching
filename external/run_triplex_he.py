#!/usr/bin/env python
"""He Stage 4 -- TRIPLEX (triplex_lopo_833_e20_ckpt, final.pt) on He et al. 2020 sections.

Same model and branches as run_triplex.py (the Visium driver), fed exactly as the port feeds her2st:
  target     224 px crop around the spot of the her2st-scale He JPEG, white pad (BLEEP's crop),
             ToTensor + ImageNet Normalize
  global     CIGAR ResNet18 512-d feature of every spot's crop (+ position -> APEG).
             Cached under ext/he/features/triplex/<SEC>.npz
  neighbour  triplex.build_features._build_neighbor: exact array-grid lookup of the 5x5 offsets
             around each spot, centre = token 12 -- identical to her2st (He is the same platform;
             the Visium nearest-spot workaround is not needed)
  position   (array_col, array_row) = He (x, y), as her2st
  truth      BLEEP/TRIPLEX target: panel counts -> CPM over the panel -> natural log1p
  APEG grid  recovered per fold exactly as run_triplex.py (--grid scan: the (W, H) that reproduces
             the stored held-out predictions; 'modal' or 'W,H' also accepted)

The her2st held-out sections for the paired comparison come from run_triplex.py (runbook G.1).

writes: <out>/fold0<k>_<P>/preds/<SEC>.npz   (he_common.write_preds, cohort 'he')
usage:  cd /workspace/st_benching && python external/run_triplex_he.py [--folds A,B] [--sections ...]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import he_common as H  # noqa: E402
import run_triplex as R  # noqa: E402  (BD with read_counts patch, grid helpers, run_model)
from run_triplex import BD, TC, TH, TAG, TriTestSections, _load_section, build_cigar_encoder, encode_patches  # noqa: E402
from triplex.build_features import _build_neighbor  # noqa: E402

FEAT_HE = H.HE / "features" / "triplex"
_find_image_her2st = BD.find_image


def _find_image_any(root, section):
    """He sections live under ST-imgs/<patient id>/<SEC>/, not ST-imgs/<SEC[0]>/<SEC>/."""
    hits = sorted(glob.glob(os.path.join(root, "ST-imgs", "*", section, "*.jpg*")))
    return hits[0] if hits else _find_image_her2st(root, section)


BD.find_image = _find_image_any


def he_inputs(sec, panel_cache, device, enc):
    """Patches, CIGAR global features, neighbour tensor, positions, truth -- built once."""
    q = BD.Her2stCLIPDataset(str(H.HE_DATA), [sec], panel_cache, is_train=False, verbose=False)
    S = q.sections[0]
    sid = [str(s) for s in S.spot_ids]
    patches = np.stack([S.patch(i) for i in range(len(sid))])            # (N,224,224,3) uint8
    f = FEAT_HE / f"{sec}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        assert [str(s) for s in z["spot_id"]] == sid
        glob_feat = z["feat"]
    else:
        enc = enc or build_cigar_encoder(device=device)
        glob_feat = encode_patches(enc, patches, device=device).astype(np.float32)
        np.savez(f, feat=glob_feat, spot_id=np.array(sid))
    sp = K.read_spots(sec, H.HE_DATA).loc[sid]
    array_rc = sp[["y", "x"]].to_numpy(np.int64)                         # (row, col)
    neigh, mask = _build_neighbor(glob_feat, array_rc)
    return enc, dict(patches=patches, glob=glob_feat, neigh=neigh, mask=mask,
                     pos=sp[["x", "y"]].to_numpy(np.float32), truth=q.expression_matrix(), sid=sid,
                     n_neigh=float(mask.sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=None, help="default: every exported He section")
    ap.add_argument("--subtypes", nargs="*", default=None)
    ap.add_argument("--out", default="/workspace/runs/he_triplex")
    ap.add_argument("--grid", default="scan",
                    help="'scan' (recover per fold from stored preds; default), 'modal', or 'W,H'")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(str(K.ST_BENCH))                  # upstream init looks for ./weights/cigar (run.sh)
    panel = K.load_panel()
    panel_cache = TH.load_panel()               # expression.npy / model column order
    assert set(panel_cache) == set(panel)
    perm = [panel_cache.index(g) for g in panel]

    exported = {f.name[len("counts_"):-len(".npz")] for f in H.HE_CALIB.glob("counts_*.npz")}
    secs = a.sections or [s for s in H.he_sections(subtypes=a.subtypes) if s in exported]
    missing = [s for s in secs if s not in exported]
    if missing:
        raise SystemExit(f"not exported yet: {missing} -- run export_he_like.py")
    FEAT_HE.mkdir(parents=True, exist_ok=True)
    t0, enc, he = time.time(), None, {}
    for sec in secs:
        enc, he[sec] = he_inputs(sec, panel_cache, device, enc)
    del enc
    torch.cuda.empty_cache()
    nn_ = [v["n_neigh"] for v in he.values()]
    print(f"built {len(he)} He sections in {time.time() - t0:.0f}s; neighbours/spot "
          f"{np.mean(nn_):.1f} [{np.min(nn_):.1f}, {np.max(nn_):.1f}] (her2st interior = 25)", flush=True)

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        model, meta = R.load_fold(TAG, P, device=device)
        tr = {s: _load_section(s, panel_cache) for s in fd["train"]}
        modal, spread = R.pin_grid(model, [tr[s]["coords"] for s in fd["train"]])
        if a.grid == "scan":
            te0 = TriTestSections(fd["test"][:1], panel_cache)
            st0 = np.load(os.path.join(TC.OUTPUT_DIR, TAG, f"fold_{P}", "preds", f"{fd['test'][0]}.npz"),
                          allow_pickle=True)["pred"]
            grid, rows = R.scan_grid(model, te0, fd["test"][0], st0, device)
            print(f"  [{P}] grid scan: " + ", ".join(f"{g}:{d:.2g}" for g, d in rows[:4]), flush=True)
            if rows[0][1] > 1e-2:
                print(f"  [{P}] WARNING: no grid reproduces the stored predictions (best {rows[0][1]:.3g})",
                      flush=True)
            del te0
        elif a.grid == "modal":
            grid = modal
        else:
            grid = tuple(int(v) for v in a.grid.split(","))
        R.set_grid(model, grid)
        trainmean = np.concatenate([tr[s]["expr"] for s in fd["train"]]).mean(0)[perm]
        extra = dict(ckpt=os.path.join(TC.CKPT_DIR, TAG, f"fold_{P}", "final.pt"), cohort="he",
                     apeg_grid=grid, apeg_grid_rule=a.grid, apeg_grid_modal=modal,
                     neighbours="exact array-grid lookup (as her2st)", train=fd["train"])
        out_dir = os.path.join(out_root, f"fold0{k}_{P}")
        pcc = []
        for sec, v in he.items():
            pred = R.run_model(model, v["patches"], v["neigh"], v["mask"], v["glob"], v["pos"], device)
            H.write_preds(out_dir, sec, pred=pred[:, perm], truth=v["truth"][:, perm], spot_ids=v["sid"],
                          genes=panel, trainmean=trainmean, model="triplex", fold=k, extra=extra)
            pcc.append(np.nanmean(K.per_gene_pcc(pred, v["truth"])))
            torch.cuda.empty_cache()
        print(f"fold {P}: APEG grid {grid}, {len(he)} He sections, mean PCC {np.mean(pcc):.4f} "
              f"[{np.min(pcc):.4f}, {np.max(pcc):.4f}], {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
