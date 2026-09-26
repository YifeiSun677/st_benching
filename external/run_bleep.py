#!/usr/bin/env python
"""Stage 5 driver -- BLEEP (bleep_lopo_e10 weights), her2st held-out + Visium.

For each fold P (held-out her2st patient):
  reference = the fold's 7 training patients from /workspace/her2st_cache, embedded
              with the EXPRESSION encoder (exactly as infer_bleep.py)
  queries   = P's own her2st sections (round-trip, Stage 4) and I1 I2 J1 K1,
              all through the on-the-fly Her2stCLIPDataset (224 px crops, white pad)
  pred      = mean expression of the top-50 retrieved reference spots ('average')
  truth     = BLEEP's own target: panel-aligned counts -> CPM -> natural log1p
  trainmean = mean of the reference expression (= 'baseline' in the stored preds)

usage: cd /workspace/st_benching && python external/run_bleep.py --folds all
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K

sys.path.insert(0, str(K.ST_BENCH / "bleep"))
import her2st_dataset as HD  # noqa: E402
from her2st_dataset import Her2stCLIPDataset, align_to_panel, cpm_log1p  # noqa: E402


def _read_counts_any(root, section):
    """The port hard-codes ST-cnts/<sec>.tsv.gz; the pod's her2st copy may hold plain .tsv.
    Accept either, same parsing (tab-separated, first column = 'XxY' spot id)."""
    import pandas as pd
    for ext in (".tsv.gz", ".tsv"):
        p = os.path.join(root, "ST-cnts", f"{section}{ext}")
        if os.path.exists(p):
            return pd.read_csv(p, sep="\t", index_col=0)
    raise FileNotFoundError(f"no ST-cnts/{section}.tsv[.gz] under {root}")


HD.read_counts = _read_counts_any       # Her2stSection looks read_counts up in HD at call time
read_counts = _read_counts_any
from infer_bleep import embed  # noqa: E402
from models import CLIPModel  # noqa: E402
from patch_cache import CachedCLIPDataset  # noqa: E402


def per_gene_pcc(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all", help="all or e.g. B or A,B")
    ap.add_argument("--sections", nargs="*", default=K.VISIUM_SECTIONS)
    ap.add_argument("--ckpt_root", default="/workspace/runs/bleep_lopo_e10")
    ap.add_argument("--cache", default="/workspace/her2st_cache")
    ap.add_argument("--out", default="/workspace/runs/ext_bleep")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--skip_roundtrip", action="store_true")
    ap.add_argument("--rt_tol", type=float, default=1e-3,
                    help="max |PCC_new - PCC_stored| per section. Retrieval is discrete: a tiny embedding "
                         "change swaps one of the top-50 neighbours, so exact pred equality is not expected")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    panel = K.load_panel()
    folds = K.lopo_folds()
    if a.folds != "all":
        want = a.folds.split(",")
        folds = [f for f in folds if f["patient"] in want]
    rt_rows = []
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        ckpt = os.path.join(a.ckpt_root, P, "last.pt")
        out_dir = os.path.join(a.out, f"fold0{k}_{P}")
        model = CLIPModel().to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.eval()

        ref = CachedCLIPDataset(a.cache, fd["train"], panel, is_train=False)
        ref_emb = F.normalize(torch.from_numpy(
            embed(model, ref, dev, a.batch_size, a.num_workers, "spot")), dim=-1)
        ref_expr = ref.expression_matrix()
        trainmean = ref_expr.mean(0)

        queries = ([] if a.skip_roundtrip else fd["test"]) + list(a.sections)
        for sec in queries:
            root = K.VIS_ROOT if sec in K.VISIUM_SECTIONS else K.HER2ST_ROOT
            q = Her2stCLIPDataset(str(root), [sec], panel, is_train=False, verbose=False)
            q_emb = F.normalize(torch.from_numpy(
                embed(model, q, dev, a.batch_size, a.num_workers, "image")), dim=-1)
            kk = min(a.top_k, len(ref_emb))
            idx = torch.topk(q_emb @ ref_emb.T, k=kk, dim=-1).indices.numpy()
            pred = ref_expr[idx].mean(axis=1)
            truth = q.expression_matrix()
            spot_ids = q.sections[0].spot_ids

            truth_ps = centres = None
            if sec in K.VISIUM_SECTIONS:
                cnt = read_counts(str(root), sec).loc[spot_ids]
                raw, _ = align_to_panel(cnt, panel)
                agg, centres = K.aggregate_counts(raw.values.astype(np.float32), spot_ids, sec)
                truth_ps = cpm_log1p(agg.astype(np.float32))

            K.write_preds(out_dir, sec, pred=pred, truth=truth, spot_ids=spot_ids, genes=panel,
                          trainmean=trainmean, truth_ps=truth_ps, centre_ids=centres,
                          model="bleep", fold=k,
                          extra=dict(ckpt=ckpt, top_k=kk, method="average", cache=a.cache,
                                     train=fd["train"], test=fd["test"], visium=list(a.sections)))

            if sec not in K.VISIUM_SECTIONS:          # Stage 4a: compare with stored LOPO preds
                z = np.load(os.path.join(a.ckpt_root, P, "preds.npz"), allow_pickle=True)
                keys = [str(x) for x in z["query_keys"]]
                row = {kk_: i for i, kk_ in enumerate(keys)}
                sel = [row[f"{sec}:{s}"] for s in spot_ids]
                sp, st = z["pred"][sel], z["truth"][sel]
                new_pcc = np.nanmean(per_gene_pcc(pred, truth))
                old_pcc = np.nanmean(per_gene_pcc(sp, st))
                rt_rows.append(dict(model="bleep", fold=P, section=sec,
                                    max_abs_pred_diff=float(np.abs(pred - sp).max()),
                                    max_abs_truth_diff=float(np.abs(truth - st).max()),
                                    pcc_stored=round(float(old_pcc), 4), pcc_new=round(float(new_pcc), 4)))
                print("  roundtrip", rt_rows[-1])
        print(f"fold {P}: done in {time.time()-t0:.0f}s")

    if rt_rows:
        import pandas as pd
        p = K.EXT / "roundtrip.tsv"
        df = pd.DataFrame(rt_rows)
        if p.exists():
            old = pd.read_csv(p, sep="\t")
            df = pd.concat([old[old.model != "bleep"], df])
        df.to_csv(p, sep="\t", index=False)
        bad = df[(df.model == "bleep") & ((df.pcc_new - df.pcc_stored).abs() > a.rt_tol)]
        print("ROUNDTRIP", "FAIL" if len(bad) else "PASS", f"({len(bad)} sections off by >{a.rt_tol:g})")


if __name__ == "__main__":
    main()
