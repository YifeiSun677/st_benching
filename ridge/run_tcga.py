"""
ridge/run_tcga.py -- the her2st ridge baseline (UNI, local) on TCGA-BRCA windows (bulk transfer arm).

GPU for the features only: UNI v1 on 112-um crops of the her2st-scale window JPEG resized to 224 px -- the
same crop / to_uni_input / embed as external/run_stflow_tcga.py (whose features were not cached).  Each
section is embedded once and the 8 her2st ridge folds (ridge.external_fit) are applied to it on the CPU.
--cache DIR keeps the embeddings (~1.8 GB for 450k spots) so a rerun skips the GPU.

  raw  ridge output on target cp10k_panel_ln = log1p(CP10K over the panel), panel order (as STFlow's 'raw')
  lin  max(expm1(raw), 0)                     -> tcga/pseudobulk.py sums it per patient (panel CPM)

writes  <out>/fold0<k>_<P>/agg/<SEC>.npz  (+ spots/ for the default spot patients), external/tcga_common.write_agg
then    python tcga/pseudobulk.py --runs <out> --out /workspace/results/tcga
        python tcga/score_tcga.py --pb /workspace/results/tcga/pseudobulk_ridge_uni.tsv.gz

usage   cd /workspace/st_benching && python -m ridge.run_tcga --verify [--limit 6] [--cache /workspace/ext/tcga/features/uni_hest112]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import external_fit as EF

EXT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external")
sys.path.insert(0, EXT)
import common as K  # noqa: E402
import tcga_common as T  # noqa: E402

MODEL = "ridge_uni"


def embedder(device):
    """UNI + the STFlow port's crop / input / embed (the exact functions the her2st features came from)."""
    import run_stflow as R  # noqa: F401  (puts the stflow port on sys.path)
    from run_stflow import BF, SC
    from run_stflow_tcga import uni_features
    uni = BF.load_uni(device)
    ref = json.loads((K.CALIB / "her2st_scale.json").read_text())
    crop_px = int(round(SC.HEST_PATCH_UM / ref["um_per_px_ref"]))
    return (lambda path, px, py: uni_features(uni, path, px, py, crop_px, device)), crop_px


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--batch", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spots-for", nargs="*", default=None)
    ap.add_argument("--out", default="/workspace/runs/tcga_ridge_uni")
    ap.add_argument("--cache", default=None, help="dir to keep / reuse UNI embeddings per section")
    ap.add_argument("--verify", action="store_true", help="first check the refitted folds on her2st")
    a = ap.parse_args()

    folds, meta, cache = EF.fit_folds()
    if a.verify and not EF.verify(folds, meta, cache):
        raise SystemExit("verify FAILED: the refitted ridge does not reproduce the her2st run")
    panel = meta["genes"]
    assert panel == K.load_panel(), "ridge panel order != external panel order"
    del cache

    w = T.select_sections(a.sections, a.patients, a.kinds, a.batch, a.limit)
    spot_pats = set(T.default_spot_patients() if a.spots_for is None else a.spots_for)
    dirs = {P: os.path.join(a.out, f"fold0{f['fold']}_{P}") for P, f in folds.items()}
    cache_dir = Path(a.cache) if a.cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    embed, crop_px = None, None
    print(f"{len(w)} sections, {int(w.n_spots.sum())} spots; 8 ridge folds ({meta['source']}, {meta['target']})")
    t_all, n_all = time.time(), 0
    for row in w.itertuples():
        if all(T.agg_path(d, row.section).exists() for d in dirs.values()):
            continue
        t0 = time.time()
        sp = T.read_spots(row.section)
        cf = cache_dir / f"{row.section}.npz" if cache_dir else None
        if cf and cf.exists():
            z = np.load(cf, allow_pickle=True)
            assert [str(s) for s in z["spot_id"]] == list(sp.index), f"{row.section}: cached rows != spots"
            feat = z["feat"]
        else:
            if embed is None:
                import torch
                embed, crop_px = embedder("cuda" if torch.cuda.is_available() else "cpu")
                print(f"UNI loaded; crop {crop_px}px = 112 um", flush=True)
            feat = embed(T.image_path(row.section), sp.pixel_x.values, sp.pixel_y.values)
            if cf:
                np.savez(cf, feat=feat.astype(np.float32), spot_id=np.array(sp.index), crop_px=crop_px)
        assert feat.shape == (len(sp), meta["dim"]), (row.section, feat.shape)
        for P, f in folds.items():
            raw = EF.predict(f, feat)
            lin = np.maximum(np.expm1(raw), 0.0)
            T.write_agg(dirs[P], row, pred_raw=raw, pred_lin=lin, spot_ids=list(sp.index), genes=panel,
                        model=MODEL, fold=f["fold"], inverse="max(expm1(raw), 0)",
                        save_spots=row.patient in spot_pats,
                        extra=dict(meta={k: v for k, v in meta.items() if k != "genes"}, alpha=f["alpha"],
                                   cohort="tcga", raw="log1p(CP10K over the panel)", inverse="max(expm1(raw), 0)"))
        n_all += len(sp)
        print(f"{row.section}: {len(sp)} spots, {time.time() - t0:.1f}s", flush=True)
    print(f"DONE {n_all} spots in {time.time() - t_all:.0f}s -> {a.out}\n"
          f"next: python tcga/pseudobulk.py --runs {a.out} --out /workspace/results/tcga")


if __name__ == "__main__":
    main()
