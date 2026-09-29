#!/usr/bin/env python
"""He Stage 4 -- BLEEP (bleep_lopo_e10 weights) on He et al. 2020 sections.

Same model, retrieval and target as run_bleep.py (the Visium driver); only the cohort differs:
  checkpoint  /workspace/runs/bleep_lopo_e10/<P>/last.pt
  reference   the fold's 7 training patients from /workspace/her2st_cache, embedded with the
              EXPRESSION encoder (as infer_bleep.py); trainmean = mean reference expression
  queries     He sections from ext/he/her2st_like/data through the port's Her2stCLIPDataset
              (224 px crops of the her2st-scale JPEG, white pad at edges)
  pred        mean expression of the top-50 retrieved reference spots ('average')
  truth       BLEEP's own target: panel-aligned counts -> CPM -> natural log1p (her2st_dataset)

Every He patient is unseen by all 8 fold models.  Sections are the outer loop: each He image is
loaded once and embedded by all 8 image encoders.  The her2st held-out sections for the paired
comparison come from run_bleep.py (runbook B.1), written into the same --out tree.

writes: <out>/fold0<k>_<P>/preds/<SEC>.npz   (he_common.write_preds, cohort 'he')
usage:  cd /workspace/st_benching && python external/run_bleep_he.py [--folds A,B] [--sections BC23287_C1 ...]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import he_common as H  # noqa: E402
import run_bleep as R  # noqa: E402  (patches HD.read_counts; brings embed, CLIPModel, CachedCLIPDataset)
from run_bleep import HD, CLIPModel, CachedCLIPDataset, embed  # noqa: E402

_find_image_her2st = HD.find_image


def _find_image_any(root, section):
    """He sections live under ST-imgs/<patient id>/<SEC>/, not ST-imgs/<SEC[0]>/<SEC>/."""
    hits = sorted(glob.glob(os.path.join(root, "ST-imgs", "*", section, "*.jpg*")))
    return hits[0] if hits else _find_image_her2st(root, section)


HD.find_image = _find_image_any         # Her2stSection looks find_image up in HD at call time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all", help="all or e.g. B or A,B")
    ap.add_argument("--sections", nargs="*", default=None, help="default: every exported He section")
    ap.add_argument("--subtypes", nargs="*", default=None)
    ap.add_argument("--ckpt_root", default="/workspace/runs/bleep_lopo_e10")
    ap.add_argument("--cache", default="/workspace/her2st_cache")
    ap.add_argument("--out", default="/workspace/runs/he_bleep")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    panel = K.load_panel()

    # ---- all fold models + their reference banks ---------------------------------------------
    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    models = {}
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        ckpt = os.path.join(a.ckpt_root, P, "last.pt")
        model = CLIPModel().to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.eval()
        ref = CachedCLIPDataset(a.cache, fd["train"], panel, is_train=False)
        ref_emb = F.normalize(torch.from_numpy(
            embed(model, ref, dev, a.batch_size, a.num_workers, "spot")), dim=-1)
        ref_expr = ref.expression_matrix()
        models[P] = dict(fold=k, model=model, ref_emb=ref_emb, ref_expr=ref_expr, trainmean=ref_expr.mean(0),
                         extra=dict(ckpt=ckpt, top_k=min(a.top_k, len(ref_emb)), method="average",
                                    cache=a.cache, train=fd["train"], cohort="he",
                                    target="panel counts -> CPM -> log1p (her2st_dataset.cpm_log1p)"))
        print(f"fold {P}: reference {len(ref_emb)} spots embedded in {time.time() - t0:.0f}s", flush=True)

    exported = {os.path.basename(f)[len("counts_"):-len(".npz")] for f in glob.glob(str(H.HE_CALIB / "counts_*.npz"))}
    secs = a.sections or [s for s in H.he_sections(subtypes=a.subtypes) if s in exported]
    missing = [s for s in secs if s not in exported]
    if missing:
        raise SystemExit(f"not exported yet: {missing} -- run export_he_like.py")

    for sec in secs:
        t0 = time.time()
        q = HD.Her2stCLIPDataset(str(H.HE_DATA), [sec], panel, is_train=False, verbose=False)
        truth = q.expression_matrix()
        spot_ids = q.sections[0].spot_ids
        pcc = []
        for P, m in models.items():
            q_emb = F.normalize(torch.from_numpy(
                embed(m["model"], q, dev, a.batch_size, a.num_workers, "image")), dim=-1)
            kk = min(a.top_k, len(m["ref_emb"]))
            idx = torch.topk(q_emb @ m["ref_emb"].T, k=kk, dim=-1).indices.numpy()
            pred = m["ref_expr"][idx].mean(axis=1)
            H.write_preds(os.path.join(out_root, f"fold0{m['fold']}_{P}"), sec, pred=pred, truth=truth,
                          spot_ids=spot_ids, genes=panel, trainmean=m["trainmean"], model="bleep",
                          fold=m["fold"], extra=m["extra"])
            pcc.append(np.nanmean(R.per_gene_pcc(pred, truth)))
        print(f"{sec}: {len(spot_ids)} spots, {q.sections[0].n_padded} edge-padded, "
              f"mean PCC over folds {np.mean(pcc):.4f} [{np.min(pcc):.4f}, {np.max(pcc):.4f}]  "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
