#!/usr/bin/env python
"""Stage 5 driver -- Path2Space (path2space_lopo_833_ckpt: 7 ik x 7 il MLPs per fold).

Per fold P:
  features  frozen CTransPath (768-d) on 224 px tiles at round(pixel), zero pad, per-tile
            Macenko to the companion's fixed target (raw tile if Macenko fails) -- the port's
            own build_features._crop + p2s_import extractor/normaliser.  Visium features are
            cached once under /workspace/ext/features/path2space/.
  ensemble  ik_0..ik_6.pt from /workspace/p2s_ckpt/<tag>/fold_P; mean over il inside each
            ik, then mean over ik -- the same nested order as run_lopo.py
  truth     the port's dataset.transform_counts ('lognorm'): panel counts, per-spot library
            over the PANEL, x median panel library of THE SECTION, natural log1p.  Applied
            per section, so pseudo-spots use the pseudo-spots' own median.
  round-trip  P's her2st sections from the existing feature cache vs the stored
            results/<tag>/preds/<sec>.npz (same weights: expect an exact match)
  primary footing is RAW (no KDTree smoothing), as in the main table

usage: cd /workspace/st_benching && python external/run_path2space.py --folds B
needs: pip install spams-bin opencv-python-headless   (Macenko; only for Visium features)
Macenko runs in a process pool (--workers, default = CPUs - 2) and prints progress every
500 tiles; features are cached, so an interrupted section restarts only that section.
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
from path2space.config import CKPT_DIR, CTRANSPATH, FEATURE_DIR, OUT_DIR, PATCH_PX  # noqa: E402
from path2space.dataset import load_section, transform_counts  # noqa: E402
from path2space.run_lopo import load_ik, make_ik_subsets, sections_by_patient  # noqa: E402
from path2space.train_mlp import predict  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
TAG = "path2space_lopo_833_ckpt"
FEAT_EXT = K.EXT / "features" / "path2space"


def per_gene_pcc(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


_G = {}     # per-process globals for the Macenko pool (filled before fork, or in the initializer)


def _pool_init():
    from path2space.p2s_import import macenko_normalizer
    _G["norm"] = macenko_normalizer()


def _one_tile(i):
    """Crop + QC flag + Macenko for tile i -- identical to build_features.build()'s loop body."""
    from path2space.build_features import _crop
    from path2space.p2s_import import evaluate_tile
    tile = _crop(_G["img"], int(_G["px"][i]), int(_G["py"][i]), _G["r"])
    flag = int(evaluate_tile(tile, 15, 0.5))
    try:
        return _G["norm"].transform(tile), flag, 0
    except Exception:
        return tile, flag, 1


def visium_features(sec, panel, workers):
    """Mirror of path2space.build_features.build() for one Visium section, cached.
    Macenko is per-tile and independent, so it is spread over `workers` processes;
    tile order and every per-tile operation are unchanged."""
    import multiprocessing as mp
    FEAT_EXT.mkdir(parents=True, exist_ok=True)
    f = FEAT_EXT / f"{sec}.npz"
    cnt = K.read_counts(sec, K.VIS_ROOT)
    sp = K.read_spots(sec, K.VIS_ROOT)
    sp = sp[sp.index.isin(cnt.index)]                    # spot-file order, as build() does
    if f.exists():
        z = np.load(f, allow_pickle=True)
        assert list(z["spot_id"]) == list(sp.index)
        return z["feat"], cnt.loc[sp.index]
    from path2space.p2s_import import CTransPathExtractor
    d = K.VIS_ROOT / "ST-imgs" / sec[0] / sec
    _G["img"] = np.asarray(Image.open(d / sorted(os.listdir(d))[0]).convert("RGB"))
    _G["px"] = np.round(sp.pixel_x.values).astype(int)
    _G["py"] = np.round(sp.pixel_y.values).astype(int)
    _G["r"] = PATCH_PX // 2
    n = len(sp)
    tiles, flags, n_fail = [None] * n, [0] * n, 0
    t0 = time.time()
    ctx = mp.get_context("fork")                         # children inherit _G (image, coords)
    with ctx.Pool(workers, initializer=_pool_init) as pool:
        for i, (normed, flag, fail) in enumerate(pool.imap(_one_tile, range(n), chunksize=32)):
            tiles[i], flags[i], n_fail = Image.fromarray(normed), flag, n_fail + fail
            if (i + 1) % 500 == 0 or i + 1 == n:
                el = time.time() - t0
                print(f"  {sec}: Macenko {i+1}/{n} tiles, {el:.0f}s elapsed, "
                      f"~{el/(i+1)*(n-i-1):.0f}s left", flush=True)
    del _G["img"]
    feat = CTransPathExtractor(str(CTRANSPATH)).extract(tiles).astype(np.float32)
    np.savez(f, feat=feat, spot_id=np.array(sp.index), select=np.array(flags, np.int8))
    print(f"  {sec}: CTransPath features {feat.shape}, Macenko failed on {n_fail} tiles, "
          f"{time.time()-t0:.0f}s total", flush=True)
    return feat, cnt.loc[sp.index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=K.VISIUM_SECTIONS)
    ap.add_argument("--out", default="/workspace/runs/ext_path2space")
    ap.add_argument("--skip_roundtrip", action="store_true")
    ap.add_argument("--rt_tol", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="processes for per-tile Macenko (Visium features only)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    panel = K.load_panel()
    sbp = sections_by_patient()
    stored_dir = OUT_DIR.parent / TAG / "preds"

    vis = {}
    for sec in a.sections:
        feat, cnt = visium_features(sec, panel, a.workers)
        raw = cnt.reindex(columns=panel, fill_value=0).to_numpy(np.float32)
        truth = transform_counts(raw)
        agg, centres = K.aggregate_counts(raw.astype(np.float64), list(cnt.index), sec)
        vis[sec] = (feat, truth, transform_counts(agg.astype(np.float32)), list(cnt.index), centres)

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    rt = []
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        train_pat = [p for p in sorted(sbp) if p != P]
        subsets = make_ik_subsets(train_pat, None)
        ens = []
        for ik, subset in enumerate(subsets):
            models, meta = load_ik(CKPT_DIR / TAG / f"fold_{P}" / f"ik_{ik}.pt", dev)
            assert meta["train_patients"] == subset and meta["held_out"] == P, f"ik_{ik} metadata mismatch"
            ens.append(models)

        def run(x):
            acc = np.zeros((len(x), len(panel)))
            for models in ens:                                   # mean over il, then over ik
                acc += np.mean([predict(m, x, dev) for m in models], axis=0)
            return acc / len(ens)

        ytr = np.concatenate([transform_counts(load_section(s)["counts833"].astype(np.float32))
                              for p in train_pat for s in sbp[p]])
        trainmean = ytr.mean(0)
        out_dir = os.path.join(a.out, f"fold0{k}_{P}")
        extra = dict(ckpt=str(CKPT_DIR / TAG / f"fold_{P}"), n_ik=len(ens), n_il=len(ens[0]),
                     target="lognorm: panel library, x section median, log1p", smoothing="none",
                     train=fd["train"], test=fd["test"], visium=list(a.sections))

        if not a.skip_roundtrip:
            for sec in sbp[P]:
                d = load_section(sec)
                pred = run(d["feat"].astype(np.float32))
                truth = transform_counts(d["counts833"].astype(np.float32))
                sid = [str(s) for s in d["spot_id"]]
                K.write_preds(out_dir, sec, pred=pred, truth=truth, spot_ids=sid, genes=panel,
                              trainmean=trainmean, model="path2space", fold=k, extra=extra)
                st = np.load(stored_dir / f"{sec}.npz", allow_pickle=True)
                assert [str(s) for s in st["spot_id"]] == sid
                rt.append(dict(model="path2space", fold=P, section=sec,
                               max_abs_pred_diff=float(np.abs(pred - st["pred"]).max()),
                               max_abs_truth_diff=float(np.abs(truth - st["truth"]).max()),
                               pcc_stored=round(float(np.nanmean(per_gene_pcc(st["pred"], st["truth"]))), 4),
                               pcc_new=round(float(np.nanmean(per_gene_pcc(pred, truth))), 4)))
                print("  roundtrip", rt[-1])

        for sec, (feat, truth, truth_ps, sid, centres) in vis.items():
            K.write_preds(out_dir, sec, pred=run(feat), truth=truth, spot_ids=sid, genes=panel,
                          trainmean=trainmean, truth_ps=truth_ps, centre_ids=centres,
                          model="path2space", fold=k, extra=extra)
        print(f"fold {P}: {len(ens)}x{len(ens[0])} MLPs, done in {time.time()-t0:.0f}s")

    if rt:
        p = K.EXT / "roundtrip.tsv"
        cur = pd.DataFrame(rt)
        df = cur
        if p.exists():
            old = pd.read_csv(p, sep="\t")
            df = pd.concat([old[~((old.model == "path2space") & old.fold.isin(cur.fold))], cur])
        df.to_csv(p, sep="\t", index=False)
        bad = cur[(cur.pcc_new - cur.pcc_stored).abs() > a.rt_tol]
        print("ROUNDTRIP", "FAIL" if len(bad) else "PASS", f"({len(bad)} sections off by >{a.rt_tol:g})")


if __name__ == "__main__":
    main()
