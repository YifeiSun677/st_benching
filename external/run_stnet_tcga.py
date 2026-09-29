#!/usr/bin/env python
"""TCGA stage B -- ST-Net (densenet121_224/top_833, epoch 25) on TCGA-BRCA whole-slide windows.

Same model and input as run_stnet_he.py; only the cohort and the output differ:
  checkpoint  /workspace/ST-Net/output/densenet121_224/top_833/<P>_checkpoints/epoch_25.pt
  input       224 px crop centred on round(pixel) of the her2st-scale window JPEG,
              ToTensor + Normalize(mean, std) parsed from <P>_gene.log (per fold)  -- run_stnet.crop_all
              / forward, exactly as for He and Visium
  output      raw = log((1 + c) / (n + Z)) per panel gene (model columns reordered to the panel)
              lin = exp(raw)  -- ST-Net's own inverse: the spot's share of the her2st gene universe
  no truth    ST-Net needs no counts as input; TCGA has no spot truth.  ST-cnts are never read.

Every TCGA patient is unseen by all 8 folds, so each fold predicts every section.  Sections are the
outer loop: each is cropped once and pushed through all 8 models.  Resumable: a section whose
agg file exists for every fold is skipped.

--he-check SEC   runs one He section through THIS script's code path and compares with the stored
                 He predictions (/workspace/runs/he_stnet/fold0<k>_<P>/preds/<SEC>.npz): the TCGA
                 path must reproduce them (smoke test B.3.3).

writes: <out>/fold0<k>_<P>/agg/<SEC>.npz  (+ spots/<SEC>.npz for --spots-for patients)
usage:  cd /workspace/st_benching && python external/run_stnet_tcga.py [--limit 6] [--batch 1] [--he-check BC23287_C1]
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import run_stnet as R  # noqa: E402  (import_stnet, norm_stats, load_model, forward, crop_all)
import tcga_common as T  # noqa: E402


def he_check(sec, models, perm, dev, he_root, bs):
    import he_common as H
    sp = K.read_spots(sec, H.HE_DATA)
    idir = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
    worst = 0.0
    for P, m in models.items():
        f = os.path.join(he_root, f"fold0{m['fold']}_{P}", "preds", f"{sec}.npz")
        z = np.load(f, allow_pickle=True)
        sid = [str(s) for s in z["spot_id"]]
        s = sp.loc[sid]
        patches = R.crop_all(idir / sorted(os.listdir(idir))[0], np.round(s.pixel_x.values).astype(int),
                             np.round(s.pixel_y.values).astype(int))
        pred = R.forward(m["model"], patches, m["mean"], m["std"], dev, bs=bs)[:, perm]
        d = float(np.abs(pred - z["pred"]).max())
        worst = max(worst, d)
        print(f"  he-check {sec} fold {P}: {len(sid)} spots, max |pred_tcga_path - pred_he_stored| = {d:.2e}")
    print(f"HE-CHECK {'PASS' if worst < 1e-3 else 'FAIL'} (worst {worst:.2e}, tolerance 1e-3)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--batch", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=0, help="first N sections only (smoke test)")
    ap.add_argument("--spots-for", nargs="*", default=None, help="patients whose full spot matrices are kept")
    ap.add_argument("--gene_list", default="/workspace/panels/panel_train.txt")
    ap.add_argument("--out", default="/workspace/runs/tcga_stnet")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--he-check", default=None, help="He section to verify the code path, then exit")
    ap.add_argument("--he-root", default="/workspace/runs/he_stnet")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    stnet = R.import_stnet()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = K.load_panel()

    root = stnet.config.SPATIAL_PROCESSED_ROOT
    uni_ensg = [str(g) for g in pickle.load(open(os.path.join(root, "gene.pkl"), "rb"))]
    uni_sym = [str(stnet.utils.ensembl.symbol[g]) for g in uni_ensg]
    glist = open(a.gene_list).read().split()
    out_genes = [g for g, e in zip(uni_sym, uni_ensg) if g in glist or e in glist]
    assert len(out_genes) == len(panel) and set(out_genes) == set(panel), \
        f"model outputs {len(out_genes)} genes; panel {len(panel)}; diff {sorted(set(out_genes) ^ set(panel))[:10]}"
    perm = [out_genes.index(g) for g in panel]

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    models = {}
    for fd in folds:
        P = fd["patient"]
        mean, std = R.norm_stats(P)
        models[P] = dict(fold=fd["fold"], model=R.load_model(P, len(out_genes), dev, stnet), mean=mean, std=std,
                         extra=dict(ckpt=f"{R.RUN}/{P}_checkpoints/epoch_25.pt", norm_mean=mean, norm_std=std,
                                    train=fd["train"], cohort="tcga",
                                    raw="log((1+c)/(n+Z)), n=|gene.pkl|, Z over the her2st universe",
                                    inverse="exp(raw)"))
    print(f"loaded {len(models)} fold models on {dev}")

    if a.he_check:
        he_check(a.he_check, models, perm, dev, a.he_root, a.bs)
        return

    w = T.select_sections(a.sections, a.patients, a.kinds, a.batch, a.limit)
    spot_pats = set(T.default_spot_patients() if a.spots_for is None else a.spots_for)
    print(f"{len(w)} sections, {int(w.n_spots.sum())} spots; full spot matrices kept for {sorted(spot_pats)}")
    t_all, n_all = time.time(), 0
    for row in w.itertuples():
        dirs = {P: os.path.join(out_root, f"fold0{m['fold']}_{P}") for P, m in models.items()}
        if all(T.agg_path(d, row.section).exists() for d in dirs.values()):
            continue
        t0 = time.time()
        sp = T.read_spots(row.section)
        sid = list(sp.index)
        patches = R.crop_all(T.image_path(row.section), np.round(sp.pixel_x.values).astype(int),
                             np.round(sp.pixel_y.values).astype(int))
        sds = []
        for P, m in models.items():
            raw = R.forward(m["model"], patches, m["mean"], m["std"], dev, bs=a.bs)[:, perm]
            T.write_agg(dirs[P], row, pred_raw=raw, pred_lin=np.exp(raw.astype(np.float64)), spot_ids=sid,
                        genes=panel, model="stnet", fold=m["fold"], inverse="exp(raw)",
                        save_spots=row.patient in spot_pats, extra=m["extra"])
            sds.append(float(np.median(raw.std(0))) if len(sid) > 1 else float("nan"))
        n_all += len(sid)
        print(f"{row.section}: {len(sid)} spots, median across-spot SD of raw pred {np.nanmean(sds):.4f}, "
              f"{time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t_all
    if n_all:
        print(f"DONE {n_all} spots x {len(models)} folds in {dt:.0f}s -> {1000 * dt / n_all:.2f} s per 1000 spots")


if __name__ == "__main__":
    main()
