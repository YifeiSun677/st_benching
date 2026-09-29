#!/usr/bin/env python
"""He Stage 4 -- Path2Space (path2space_lopo_833_ckpt: 7 ik x 7 il MLPs per fold) on He et al. 2020.

Same features, ensemble and target as run_path2space.py (the Visium driver); only the cohort differs:
  features  frozen CTransPath (768-d) on 224 px tiles at round(pixel) of the her2st-scale He JPEG,
            zero pad, per-tile Macenko to the companion's fixed target (raw tile if Macenko fails)
            -- the port's own build_features._crop + p2s_import extractor/normaliser, via the
            Visium driver's pool helpers.  Spot order = spot file filtered to spots with counts.
  ensemble  ik_0..ik_6.pt from /workspace/p2s_ckpt/path2space_lopo_833_ckpt/fold_P; mean over il
            inside each ik, then mean over ik
  truth     dataset.transform_counts ('lognorm'): panel counts, per-spot panel library, x median
            panel library of THE SECTION, natural log1p
  footing   RAW (no KDTree smoothing), as in the main table

Two phases so no process forks after CUDA is live (see run_path2space.macenko_tiles):
  1 CPU   Macenko tiles for every uncached section -> ext/he/features/path2space/tiles_<SEC>.npz
  2 GPU   CTransPath on those tiles -> <SEC>.npz (feat, spot_id, select); tiles file removed
Both are cached per section, so an interrupted run resumes where it stopped.
The her2st held-out sections for the paired comparison come from run_path2space.py (runbook D.1).

writes: <out>/fold0<k>_<P>/preds/<SEC>.npz   (he_common.write_preds, cohort 'he')
usage:  cd /workspace/st_benching && python external/run_path2space_he.py [--workers 48] [--folds A,B]
needs:  pip install spams-bin opencv-python-headless   (Macenko)
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # workers parallelise over tiles; no nested threads
import argparse
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import he_common as H  # noqa: E402
import run_path2space as R  # noqa: E402  (_G, _pool_init, _one_tile, container_cpus, per_gene_pcc)
from run_path2space import CKPT_DIR, CTRANSPATH, PATCH_PX, TAG  # noqa: E402
from path2space.dataset import load_section, transform_counts  # noqa: E402
from path2space.run_lopo import load_ik, make_ik_subsets, sections_by_patient  # noqa: E402
from path2space.train_mlp import predict  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
FEAT_HE = H.HE / "features" / "path2space"


def he_spots(sec):
    """Counts + spot table in spot-file order, filtered to spots with counts (as build() does)."""
    cnt = K.read_counts(sec, H.HE_DATA)
    sp = K.read_spots(sec, H.HE_DATA)
    sp = sp[sp.index.isin(cnt.index)]
    return cnt.loc[sp.index], sp


def macenko_tiles(sec, workers):
    """Phase 1 for one section (CPU only; must run before any CUDA use in this process)."""
    import multiprocessing as mp
    _, sp = he_spots(sec)
    d = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
    R._G["img"] = np.asarray(Image.open(d / sorted(os.listdir(d))[0]).convert("RGB"))
    R._G["px"] = np.round(sp.pixel_x.values).astype(int)
    R._G["py"] = np.round(sp.pixel_y.values).astype(int)
    R._G["r"] = PATCH_PX // 2
    n = len(sp)
    t0 = time.time()
    if workers <= 1:
        R._pool_init()
        res = list(map(R._one_tile, range(n)))
    else:
        with mp.get_context("fork").Pool(workers, initializer=R._pool_init) as pool:
            res = pool.map(R._one_tile, range(n), chunksize=8)
    del R._G["img"]
    tiles = np.stack([t for t, _, _ in res])
    flags = np.array([f for _, f, _ in res], np.int8)
    n_fail = sum(x for _, _, x in res)
    np.savez(FEAT_HE / f"tiles_{sec}.npz", tiles=tiles, select=flags, spot_id=np.array(list(sp.index)))
    print(f"  {sec}: Macenko {n} tiles, {n_fail} failed (raw tile kept), {time.time() - t0:.0f}s", flush=True)


def he_features(sections, workers):
    FEAT_HE.mkdir(parents=True, exist_ok=True)
    todo = [s for s in sections if not (FEAT_HE / f"{s}.npz").exists()]
    for s in todo:                                               # phase 1: no CUDA touched yet
        if not (FEAT_HE / f"tiles_{s}.npz").exists():
            macenko_tiles(s, workers)
    if todo:                                                     # phase 2: GPU
        from path2space.p2s_import import CTransPathExtractor
        ext = CTransPathExtractor(str(CTRANSPATH))
        for s in todo:
            t0 = time.time()
            z = np.load(FEAT_HE / f"tiles_{s}.npz", allow_pickle=True)
            feat = ext.extract([Image.fromarray(t) for t in z["tiles"]]).astype(np.float32)
            np.savez(FEAT_HE / f"{s}.npz", feat=feat, spot_id=z["spot_id"], select=z["select"])
            os.remove(FEAT_HE / f"tiles_{s}.npz")
            print(f"  {s}: CTransPath features {feat.shape}, {time.time() - t0:.0f}s", flush=True)
    out = {}
    for s in sections:
        cnt, sp = he_spots(s)
        z = np.load(FEAT_HE / f"{s}.npz", allow_pickle=True)
        assert list(z["spot_id"]) == list(sp.index), f"{s}: cached feature rows != spot file"
        out[s] = (z["feat"], cnt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=None, help="default: every exported He section")
    ap.add_argument("--subtypes", nargs="*", default=None)
    ap.add_argument("--out", default="/workspace/runs/he_path2space")
    ap.add_argument("--workers", type=int, default=None,
                    help="processes for per-tile Macenko (default: container CPUs - 1, max 16)")
    ap.add_argument("--features_only", action="store_true", help="stop after caching features")
    a = ap.parse_args()
    if a.workers is None:
        a.workers = max(1, min(16, R.container_cpus() - 1))
    print(f"container CPUs {R.container_cpus()}, Macenko workers {a.workers}", flush=True)
    out_root = os.path.abspath(a.out)
    panel = K.load_panel()

    exported = {f.name[len("counts_"):-len(".npz")] for f in H.HE_CALIB.glob("counts_*.npz")}
    secs = a.sections or [s for s in H.he_sections(subtypes=a.subtypes) if s in exported]
    missing = [s for s in secs if s not in exported]
    if missing:
        raise SystemExit(f"not exported yet: {missing} -- run export_he_like.py")

    feats = he_features(secs, a.workers)
    if a.features_only:
        print(f"features cached for {len(feats)} sections under {FEAT_HE}")
        return
    he = {}
    for sec in secs:
        feat, cnt = feats[sec]
        raw = cnt.reindex(columns=panel, fill_value=0).to_numpy(np.float32)
        he[sec] = (feat, transform_counts(raw), list(cnt.index))

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sbp = sections_by_patient()
    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        train_pat = [p for p in sorted(sbp) if p != P]
        ens = []
        for ik, subset in enumerate(make_ik_subsets(train_pat, None)):
            models, meta = load_ik(CKPT_DIR / TAG / f"fold_{P}" / f"ik_{ik}.pt", dev)
            assert meta["train_patients"] == subset and meta["held_out"] == P, f"ik_{ik} metadata mismatch"
            ens.append(models)

        def run(x):
            acc = np.zeros((len(x), len(panel)))
            for models in ens:                                   # mean over il, then over ik
                acc += np.mean([predict(m, x, dev) for m in models], axis=0)
            return acc / len(ens)

        trainmean = np.concatenate([transform_counts(load_section(s)["counts833"].astype(np.float32))
                                    for p in train_pat for s in sbp[p]]).mean(0)
        extra = dict(ckpt=str(CKPT_DIR / TAG / f"fold_{P}"), n_ik=len(ens), n_il=len(ens[0]), cohort="he",
                     target="lognorm: panel library, x section median, log1p", smoothing="none",
                     train=fd["train"])
        out_dir = os.path.join(out_root, f"fold0{k}_{P}")
        pcc = []
        for sec, (feat, truth, sid) in he.items():
            pred = run(feat.astype(np.float32))
            H.write_preds(out_dir, sec, pred=pred, truth=truth, spot_ids=sid, genes=panel,
                          trainmean=trainmean, model="path2space", fold=k, extra=extra)
            pcc.append(np.nanmean(R.per_gene_pcc(pred, truth)))
        print(f"fold {P}: {len(ens)}x{len(ens[0])} MLPs, {len(he)} He sections, mean PCC {np.mean(pcc):.4f} "
              f"[{np.min(pcc):.4f}, {np.max(pcc):.4f}], {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
