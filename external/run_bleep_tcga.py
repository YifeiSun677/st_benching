#!/usr/bin/env python
"""TCGA stage B -- BLEEP (bleep_lopo_e10 weights) on TCGA-BRCA whole-slide windows.

Same model, reference and retrieval as run_bleep_he.py:
  checkpoint  /workspace/runs/bleep_lopo_e10/<P>/last.pt
  reference   the fold's 7 training patients from /workspace/her2st_cache, embedded with the
              EXPRESSION encoder; reference expression = log1p(CPM over the panel)
  queries     224 px crops of the her2st-scale window JPEG, white pad at edges, ImageNet
              normalisation -- the port's Her2stSection.patch / Her2stCLIPDataset.transform
  output      raw = mean log1p(CPM) of the top-50 retrieved reference spots ('average')
              lin = expm1(raw)  -- BLEEP's own inverse (panel CPM)
  no truth    the port's dataset reads ST-cnts only to list spots and build a target; the TCGA
              placeholders keep every spot and the target is never used.  The image encoder never
              sees counts (placeholder invariance is structural).

Each section's image tensors are built once (single process, no DataLoader) and embedded by all 8
image encoders -- the same batches embed() would produce, since it iterates the dataset in order.

--he-check SEC   one He section through THIS code path vs the stored He predictions
                 (/workspace/runs/he_bleep/fold0<k>_<P>/preds/<SEC>.npz).  Retrieval is discrete: a
                 1e-6 embedding change can swap one of 50 neighbours, so PASS = >= 99 % of spots with
                 max |diff| < 1e-4 and every fold's pred-vs-stored correlation > 0.999.

writes: <out>/fold0<k>_<P>/agg/<SEC>.npz  (+ spots/<SEC>.npz for --spots-for patients)
usage:  cd /workspace/st_benching && python external/run_bleep_tcga.py [--limit 6] [--he-check BC23287_C1]
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
import run_bleep as R  # noqa: E402  (patches HD.read_counts; brings embed, CLIPModel, CachedCLIPDataset)
import tcga_common as T  # noqa: E402
from run_bleep import HD, CLIPModel, CachedCLIPDataset, embed  # noqa: E402

_find_image_her2st = HD.find_image


def _find_image_any(root, section):
    """TCGA (and He) sections live under ST-imgs/<patient id>/<SEC>/, not ST-imgs/<SEC[0]>/<SEC>/."""
    hits = sorted(glob.glob(os.path.join(root, "ST-imgs", "*", section, "*.jpg*")))
    return hits[0] if hits else _find_image_her2st(root, section)


HD.find_image = _find_image_any


@torch.no_grad()
def section_images(root, sec, panel):
    q = HD.Her2stCLIPDataset(str(root), [sec], panel, is_train=False, verbose=False)
    imgs = torch.stack([q[i]["image"] for i in range(len(q))])
    return imgs, q.sections[0].spot_ids


@torch.no_grad()
def retrieve(m, imgs, dev, bs, top_k):
    out = []
    for i in range(0, len(imgs), bs):
        out.append(m["model"].embed_image(imgs[i:i + bs].to(dev)).cpu())
    q_emb = F.normalize(torch.cat(out), dim=-1)
    kk = min(top_k, len(m["ref_emb"]))
    idx = torch.topk(q_emb @ m["ref_emb"].T, k=kk, dim=-1).indices.numpy()
    return m["ref_expr"][idx].mean(axis=1)


def he_check(sec, models, panel, dev, bs, top_k, he_root):
    import he_common as H
    imgs, sid = section_images(H.HE_DATA, sec, panel)
    pos = {s: i for i, s in enumerate(sid)}
    ok = True
    for P, m in models.items():
        z = np.load(os.path.join(he_root, f"fold0{m['fold']}_{P}", "preds", f"{sec}.npz"), allow_pickle=True)
        order = [pos[str(s)] for s in z["spot_id"]]
        pred = retrieve(m, imgs, dev, bs, top_k)[order]
        per_spot = np.abs(pred - z["pred"]).max(1)
        frac = float((per_spot < 1e-4).mean())
        c = float(np.corrcoef(pred.ravel(), z["pred"].ravel())[0, 1])
        ok = ok and frac >= 0.99 and c > 0.999
        print(f"  he-check {sec} fold {P}: {len(order)} spots, identical {100 * frac:.1f} %, "
              f"max |diff| {per_spot.max():.2e}, corr {c:.6f}")
    print(f"HE-CHECK {'PASS' if ok else 'FAIL'} (>= 99 % identical spots and corr > 0.999 per fold)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--batch", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spots-for", nargs="*", default=None)
    ap.add_argument("--ckpt_root", default="/workspace/runs/bleep_lopo_e10")
    ap.add_argument("--cache", default="/workspace/her2st_cache")
    ap.add_argument("--out", default="/workspace/runs/tcga_bleep")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--he-check", default=None)
    ap.add_argument("--he-root", default="/workspace/runs/he_bleep")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    panel = K.load_panel()

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
        ref_emb = F.normalize(torch.from_numpy(embed(model, ref, dev, a.batch_size, a.num_workers, "spot")), dim=-1)
        models[P] = dict(fold=k, model=model, ref_emb=ref_emb, ref_expr=ref.expression_matrix(),
                         extra=dict(ckpt=ckpt, top_k=min(a.top_k, len(ref_emb)), method="average",
                                    cache=a.cache, train=fd["train"], cohort="tcga",
                                    raw="mean log1p(CPM over the panel) of the top-k reference spots",
                                    inverse="expm1(raw)"))
        print(f"fold {P}: reference {len(ref_emb)} spots embedded in {time.time() - t0:.0f}s", flush=True)

    if a.he_check:
        he_check(a.he_check, models, panel, dev, a.batch_size, a.top_k, a.he_root)
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
        imgs, sid = section_images(T.TCGA_DATA, row.section, panel)
        assert len(sid) == row.n_spots, f"{row.section}: dataset kept {len(sid)} of {row.n_spots} spots"
        sds = []
        for P, m in models.items():
            raw = retrieve(m, imgs, dev, a.batch_size, a.top_k)
            T.write_agg(dirs[P], row, pred_raw=raw, pred_lin=np.expm1(raw.astype(np.float64)), spot_ids=sid,
                        genes=panel, model="bleep", fold=m["fold"], inverse="expm1(raw)",
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
