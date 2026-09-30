#!/usr/bin/env python
"""TCGA stage B -- Path2Space (path2space_lopo_833_ckpt: 7 ik x 7 il MLPs per fold) on TCGA-BRCA windows.

Same features, ensemble and output space as run_path2space_he.py:
  features  frozen CTransPath (768-d) on 224 px tiles at round(pixel) of the her2st-scale window JPEG,
            zero pad (build_features._crop), per-tile Macenko to the companion's fixed target (raw tile
            if Macenko fails) -- the port's own p2s_import normaliser / extractor
  ensemble  ik_0..ik_6.pt from /workspace/p2s_ckpt/path2space_lopo_833_ckpt/fold_P; mean over il inside
            each ik, then mean over ik
  output    raw = log1p(x / panel lib * section-median panel lib)  ('lognorm', the port's target space;
            the median scale multiplies every gene of a spot equally and cancels in the panel-CPM pseudo-bulk)
            lin = max(expm1(raw), 0)
  footing   RAW (no KDTree smoothing), as in the main table
  no truth  ST-cnts are never opened.

Pipeline (no tiles on disk: 450k tiles would be ~68 GB):
  CPU  a 'spawn' pool (safe after CUDA is live, unlike fork) Macenko-normalises one WHOLE section per task;
       at most 2 x workers sections are in flight, so RAM stays bounded
  GPU  CTransPath on the returned tiles, then the 8 fold ensembles; features are not cached
Macenko is a fixed-target transform applied tile by tile, so one-section-per-task gives the same tiles as
the He driver's one-tile-per-task pool.  Resumable per section.

--he-check SEC   one He section through THIS code path (fresh Macenko + CTransPath) vs the stored He
                 predictions (/workspace/runs/he_path2space/...): PASS = max |diff| < 1e-2 and corr > 0.9999.

writes: <out>/fold0<k>_<P>/agg/<SEC>.npz  (+ spots/<SEC>.npz for --spots-for patients)
usage:  cd /workspace/st_benching && python external/run_path2space_tcga.py [--workers 16] [--limit 6] [--he-check BC23287_C1]
needs:  pip install spams-bin opencv-python-headless   (Macenko)
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # workers parallelise over sections; no nested threads
import argparse
import sys
import time
from collections import deque

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import run_path2space as R  # noqa: E402  (container_cpus; puts the path2space port on sys.path)
import tcga_common as T  # noqa: E402
from run_path2space import CKPT_DIR, CTRANSPATH, PATCH_PX, TAG  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
_NORM = None


def _worker_init():
    global _NORM
    from path2space.p2s_import import macenko_normalizer
    _NORM = macenko_normalizer()


def macenko_section(args):
    """All tiles of one section: crop + QC flag + Macenko, identical to run_path2space._one_tile per tile."""
    img_path, px, py = args
    from path2space.build_features import _crop
    from path2space.p2s_import import evaluate_tile
    if _NORM is None:
        _worker_init()
    img = np.asarray(Image.open(img_path).convert("RGB"))
    r = PATCH_PX // 2
    tiles = np.empty((len(px), 2 * r, 2 * r, 3), np.uint8)
    flags = np.empty(len(px), np.int8)
    n_fail = 0
    for i in range(len(px)):
        tile = _crop(img, int(px[i]), int(py[i]), r)
        flags[i] = int(evaluate_tile(tile, 15, 0.5))
        try:
            tiles[i] = _NORM.transform(tile)
        except Exception:
            tiles[i] = tile
            n_fail += 1
    return tiles, flags, n_fail


def load_ensembles(folds, dev):
    from path2space.run_lopo import load_ik, make_ik_subsets, sections_by_patient
    sbp = sections_by_patient()
    ens = {}
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        train_pat = [p for p in sorted(sbp) if p != P]
        e = []
        for ik, subset in enumerate(make_ik_subsets(train_pat, None)):
            models, meta = load_ik(CKPT_DIR / TAG / f"fold_{P}" / f"ik_{ik}.pt", dev)
            assert meta["train_patients"] == subset and meta["held_out"] == P, f"ik_{ik} metadata mismatch"
            e.append(models)
        ens[P] = dict(fold=k, models=e,
                      extra=dict(ckpt=str(CKPT_DIR / TAG / f"fold_{P}"), n_ik=len(e), n_il=len(e[0]), cohort="tcga",
                                 raw="lognorm: panel library, x section median, log1p", smoothing="none",
                                 inverse="max(expm1(raw), 0)", train=fd["train"]))
    return ens


def run_ensemble(models_by_ik, x, n_genes, dev):
    from path2space.train_mlp import predict
    acc = np.zeros((len(x), n_genes))
    for models in models_by_ik:                            # mean over il, then over ik
        acc += np.mean([predict(m, x, dev) for m in models], axis=0)
    return acc / len(models_by_ik)


def he_check(sec, ext, ens, n_genes, dev, he_root):
    import he_common as H
    sp = K.read_spots(sec, H.HE_DATA)
    d = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
    img_path = d / sorted(os.listdir(d))[0]
    worst, ok, feat, sid_prev = 0.0, True, None, None
    for P, e in ens.items():
        z = np.load(os.path.join(he_root, f"fold0{e['fold']}_{P}", "preds", f"{sec}.npz"), allow_pickle=True)
        sid = [str(v) for v in z["spot_id"]]
        if sid != sid_prev:
            s = sp.loc[sid]
            tiles, _, n_fail = macenko_section((img_path, np.round(s.pixel_x.values).astype(int),
                                                np.round(s.pixel_y.values).astype(int)))
            feat = ext.extract([Image.fromarray(t) for t in tiles]).astype(np.float32)
            sid_prev = sid
        pred = run_ensemble(e["models"], feat, n_genes, dev)
        diff = float(np.abs(pred - z["pred"]).max())
        c = float(np.corrcoef(pred.ravel(), z["pred"].ravel())[0, 1])
        worst, ok = max(worst, diff), ok and c > 0.9999
        print(f"  he-check {sec} fold {P}: {len(sid)} spots, Macenko failures {n_fail}, max |diff| {diff:.2e}, corr {c:.6f}")
    print(f"HE-CHECK {'PASS' if (worst < 1e-2 and ok) else 'FAIL'} (worst {worst:.2e}; tolerance 1e-2, corr > 0.9999)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--batch", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spots-for", nargs="*", default=None)
    ap.add_argument("--out", default="/workspace/runs/tcga_path2space")
    ap.add_argument("--workers", type=int, default=None, help="Macenko processes (default: container CPUs - 1, max 16)")
    ap.add_argument("--he-check", default=None)
    ap.add_argument("--he-root", default="/workspace/runs/he_path2space")
    a = ap.parse_args()
    if a.workers is None:
        a.workers = max(1, min(16, R.container_cpus() - 1))
    out_root = os.path.abspath(a.out)
    panel = K.load_panel()

    import torch
    from path2space.p2s_import import CTransPathExtractor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    ens = load_ensembles(folds, dev)
    ext = CTransPathExtractor(str(CTRANSPATH))
    print(f"container CPUs {R.container_cpus()}, Macenko workers {a.workers}; {len(ens)} fold ensembles "
          f"({ens[next(iter(ens))]['extra']['n_ik']} x {ens[next(iter(ens))]['extra']['n_il']} MLPs) on {dev}", flush=True)

    if a.he_check:
        he_check(a.he_check, ext, ens, len(panel), dev, a.he_root)
        return

    w = T.select_sections(a.sections, a.patients, a.kinds, a.batch, a.limit)
    dirs = {P: os.path.join(out_root, f"fold0{e['fold']}_{P}") for P, e in ens.items()}
    todo = [r for r in w.itertuples() if not all(T.agg_path(d, r.section).exists() for d in dirs.values())]
    spot_pats = set(T.default_spot_patients() if a.spots_for is None else a.spots_for)
    print(f"{len(w)} sections ({len(todo)} to do), {int(w.n_spots.sum())} spots; full spot matrices kept for "
          f"{sorted(spot_pats)}", flush=True)

    import multiprocessing as mp
    spots = {}

    def task(r):
        sp = T.read_spots(r.section)
        spots[r.section] = sp
        return (str(T.image_path(r.section)), np.round(sp.pixel_x.values).astype(int),
                np.round(sp.pixel_y.values).astype(int))

    t_all, n_all = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers, initializer=_worker_init) as pool:
        it, pending = iter(todo), deque()
        for r in it:                                         # prime: at most 2 x workers sections in flight
            pending.append((r, pool.apply_async(macenko_section, (task(r),))))
            if len(pending) >= 2 * a.workers:
                break
        while pending:
            row, res = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, pool.apply_async(macenko_section, (task(nxt),))))
            t0 = time.time()
            tiles, _, n_fail = res.get()
            sp = spots.pop(row.section)
            feat = ext.extract([Image.fromarray(t) for t in tiles]).astype(np.float32)
            del tiles
            sds = []
            for P, e in ens.items():
                raw = run_ensemble(e["models"], feat, len(panel), dev)
                lin = np.maximum(np.expm1(raw.astype(np.float64)), 0.0)
                T.write_agg(dirs[P], row, pred_raw=raw, pred_lin=lin, spot_ids=list(sp.index), genes=panel,
                            model="path2space", fold=e["fold"], inverse="max(expm1(raw), 0)",
                            save_spots=row.patient in spot_pats, extra=e["extra"])
                sds.append(float(np.median(raw.std(0))) if len(sp) > 1 else float("nan"))
            n_all += len(sp)
            print(f"{row.section}: {len(sp)} spots, Macenko failures {n_fail}, median across-spot SD of raw pred "
                  f"{np.nanmean(sds):.4f}, GPU {time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t_all
    if n_all:
        print(f"DONE {n_all} spots x {len(ens)} folds in {dt:.0f}s -> {1000 * dt / n_all:.2f} s per 1000 spots")


if __name__ == "__main__":
    main()
