#!/usr/bin/env python
"""He Stage 4 -- ST-Net (densenet121_224/top_833, epoch 25) on He et al. 2020 sections.

Same model, input and target as run_stnet.py (the Visium driver); only the cohort differs:
  checkpoint  /workspace/ST-Net/output/densenet121_224/top_833/<P>_checkpoints/epoch_25.pt
  input       224 px crop centred on round(pixel) of the RESAMPLED He JPEG (her2st scale),
              ToTensor + Normalize(mean, std) parsed from <P>_gene.log (per fold)
  target      log((1 + c_g) / (n + Z)), n = |gene.pkl|, Z = spot total over the her2st universe;
              He columns are mapped into the universe by ENSG id first, then by ST-Net symbol
  output      model columns reordered to the panel

Every He patient is unseen by all 8 fold models, so each fold predicts every He section.
Sections are the outer loop: each section is cropped once and pushed through all 8 models.
The her2st held-out sections for the paired comparison come from run_stnet.py (runbook step 4.1),
written into the same --out tree.

writes: <out>/fold0<k>_<P>/preds/<SEC>.npz   (he_common.write_preds, cohort 'he')
usage:  cd /workspace/st_benching && python external/run_stnet_he.py [--folds A,B] [--sections BC23287_C1 ...]
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import he_common as H  # noqa: E402
import run_stnet as R  # noqa: E402  (import_stnet, norm_stats, load_model, forward, crop_all, stnet_target)


def universe_map(gene_ids, symbols, uni_ensg, uni_sym):
    """He feature j -> universe column: ENSG match first, symbol match second.
    Returns a sparse (n_features x n_universe) 0/1 matrix and match counts."""
    pe = {g: i for i, g in enumerate(uni_ensg)}
    ps = {g: i for i, g in enumerate(uni_sym)}
    rows, cols, by_e, by_s = [], [], 0, 0
    for j, (e, s) in enumerate(zip(gene_ids, symbols)):
        if e in pe:
            rows.append(j); cols.append(pe[e]); by_e += 1
        elif s in ps:
            rows.append(j); cols.append(ps[s]); by_s += 1
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(gene_ids), len(uni_ensg)))
    return M, by_e, by_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=None, help="default: every exported He section")
    ap.add_argument("--subtypes", nargs="*", default=None)
    ap.add_argument("--gene_list", default="/workspace/panels/panel_train.txt")
    ap.add_argument("--out", default="/workspace/runs/he_stnet")
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    stnet = R.import_stnet()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = K.load_panel()

    # ---- gene universe and column order (identical to run_stnet.py) ------------------------
    root = stnet.config.SPATIAL_PROCESSED_ROOT
    uni_ensg = [str(g) for g in pickle.load(open(os.path.join(root, "gene.pkl"), "rb"))]
    uni_sym = [str(stnet.utils.ensembl.symbol[g]) for g in uni_ensg]
    glist = open(a.gene_list).read().split()
    out_genes = [g for g, e in zip(uni_sym, uni_ensg) if g in glist or e in glist]
    assert len(out_genes) == len(panel) and set(out_genes) == set(panel), \
        f"model outputs {len(out_genes)} genes; panel {len(panel)}; diff {sorted(set(out_genes) ^ set(panel))[:10]}"
    perm = [out_genes.index(g) for g in panel]
    upos = [uni_sym.index(g) for g in panel]
    print(f"gene universe {len(uni_ensg)} (first id {uni_ensg[0]}); model outputs {len(out_genes)} genes")

    # ---- all fold models at once ------------------------------------------------------------
    stored = {P: np.load(f"{R.RUN}/{P}_25.npz", allow_pickle=True) for P in K.HER2ST_PATIENTS}
    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    models = {}
    for fd in folds:
        P = fd["patient"]
        mean, std = R.norm_stats(P)
        tm = np.concatenate([stored[Q]["counts"] for Q in K.HER2ST_PATIENTS if Q != P])[:, perm].mean(0)
        models[P] = dict(fold=fd["fold"], model=R.load_model(P, len(out_genes), dev, stnet), mean=mean, std=std,
                         trainmean=tm, extra=dict(ckpt=f"{R.RUN}/{P}_checkpoints/epoch_25.pt", norm_mean=mean,
                                                  norm_std=std, train=fd["train"], cohort="he",
                                                  target="log((1+c)/(n+Z)), n=|gene.pkl|, Z over the her2st universe"))
    print(f"loaded {len(models)} fold models on {dev}")

    exported = {f.name[len("counts_"):-len(".npz")] for f in H.HE_CALIB.glob("counts_*.npz")}
    secs = a.sections or [s for s in H.he_sections(subtypes=a.subtypes) if s in exported]
    missing = [s for s in secs if s not in exported]
    if missing:
        raise SystemExit(f"not exported yet: {missing} -- run export_he_like.py")

    for sec in secs:
        t0 = time.time()
        X, gene_ids, symbols, sid = H.load_counts_npz(sec)
        M, by_e, by_s = universe_map(gene_ids, symbols, uni_ensg, uni_sym)
        U = np.asarray((X @ M).todense(), dtype=np.float64)            # duplicates summed
        truth = R.stnet_target(U)[:, upos]
        sp = K.read_spots(sec, H.HE_DATA).loc[sid]
        px = np.round(sp["pixel_x"].values).astype(int)
        py = np.round(sp["pixel_y"].values).astype(int)
        idir = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
        patches = R.crop_all(idir / sorted(os.listdir(idir))[0], px, py)
        pcc = []
        for P, m in models.items():
            pred = R.forward(m["model"], patches, m["mean"], m["std"], dev, bs=a.bs)[:, perm]
            H.write_preds(os.path.join(out_root, f"fold0{m['fold']}_{P}"), sec, pred=pred, truth=truth,
                          spot_ids=sid, genes=panel, trainmean=m["trainmean"], model="stnet",
                          fold=m["fold"], extra=m["extra"])
            pcc.append(np.nanmean(R.per_gene_pcc(pred, truth)))
        print(f"{sec}: {len(sid)} spots, features mapped {by_e} by ENSG + {by_s} by symbol, "
              f"mean PCC over folds {np.mean(pcc):.4f} [{np.min(pcc):.4f}, {np.max(pcc):.4f}]  "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
