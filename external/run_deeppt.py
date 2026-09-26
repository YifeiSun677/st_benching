#!/usr/bin/env python
"""Stage 5 driver -- DeepPT (deeppt_833_raw, existing checkpoints), her2st held-out + Visium.

EPOCH RULE (state in methods): the existing <P>_mlp.pt is the BEST-VALIDATION-PCC epoch
(03_run_lopo.py saves only on improvement; validation patient = the patient after P), not
the last epoch the main table scores.  Both sides of the paired delta in this arm use the
same weights, and no Visium data touched epoch selection.  --mlp last would need
deeppt_resave_last.py first.

Per fold P:
  encoder   frozen ResNet50 (DeepPT_original/ResNet50_IMAGENET1K_V2.pt), 224 px crops at
            round(pixel), zero pad, ImageNet normalisation, fp16 autocast -- the port's own
            01_extract_features.build_encoder / encode and her2st_io.crop_patches
  AE        <P>_ae.pt (final weights, refit on the fold's training patients)
  MLP       <P>_mlp.pt (best-val epoch, default) or <P>_mlp_last.pt (--mlp last)
  truth     the port's 02_build_targets rule: CP10K over the FULL count row (all detected
            genes), panel subset, log10(x + 1)
  round-trip  P's her2st sections from the cached features_raw vs the run's stored predictions
            AT THE SAME EPOCH as the loaded weights (same weights: expect an exact match)

usage: cd /workspace/st_benching && python external/run_deeppt.py --folds B
"""
import argparse
import importlib.util
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

DP = str(K.ST_BENCH / "deeppt_her2st")
sys.path.insert(0, DP)
import her2st_io as io  # noqa: E402
from deeppt_models import AE, Predictor  # noqa: E402

spec = importlib.util.spec_from_file_location("deeppt_extract", os.path.join(DP, "01_extract_features.py"))
FX = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FX)
Image.MAX_IMAGE_PIXELS = None

FEAT = "/workspace/deeppt/features_raw"
TARG = "/workspace/deeppt/targets"
WEIGHTS = "/workspace/DeepPT_original/ResNet50_IMAGENET1K_V2.pt"
FEAT_EXT = K.EXT / "features" / "deeppt"


def per_gene_pcc(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def deeppt_target(cnt_full, panel):
    """02_build_targets: CP10K with library over ALL columns, then panel, log10(+1)."""
    lib = cnt_full.sum(axis=1).values.astype(np.float64)
    lib[lib == 0] = 1.0
    mat = np.zeros((len(cnt_full), len(panel)))
    present = [g for g in panel if g in cnt_full.columns]
    idx = [panel.index(g) for g in present]
    mat[:, idx] = cnt_full[present].values.astype(np.float64) / lib[:, None] * 1e4
    return np.log10(mat + 1.0)


def visium_features(sec, dev):
    FEAT_EXT.mkdir(parents=True, exist_ok=True)
    f = FEAT_EXT / f"{sec}.npz"
    cnt = K.read_counts(sec, K.VIS_ROOT)
    pos = K.read_spots(sec, K.VIS_ROOT).loc[cnt.index]
    if f.exists():
        z = np.load(f, allow_pickle=True)
        assert list(z["spot_id"]) == list(cnt.index)
        return z["feat"], cnt
    d = K.VIS_ROOT / "ST-imgs" / sec[0] / sec
    img = Image.open(d / sorted(os.listdir(d))[0]).convert("RGB")
    patches = io.crop_patches(img, pos[["pixel_x", "pixel_y"]])
    img.close()
    net = FX.build_encoder(WEIGHTS, dev)
    feat = FX.encode(patches, net, dev, 256).astype(np.float32)
    np.savez(f, feat=feat, spot_id=np.array(cnt.index))
    return feat, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=K.VISIUM_SECTIONS)
    ap.add_argument("--run", default="/workspace/deeppt/results/deeppt_833_raw")
    ap.add_argument("--mlp", choices=["best", "last"], default="best")
    ap.add_argument("--out", default="/workspace/runs/ext_deeppt")
    ap.add_argument("--skip_roundtrip", action="store_true")
    ap.add_argument("--rt_tol", type=float, default=1e-4)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    panel = K.load_panel()
    assert open(os.path.join(TARG, "genes.txt")).read().split() == panel, "targets/genes.txt != panel"

    vis = {}
    for sec in a.sections:
        feat, cnt = visium_features(sec, dev)
        truth = deeppt_target(cnt, panel)
        agg, centres = K.aggregate_counts(cnt.values.astype(np.float64), list(cnt.index), sec)
        truth_ps = deeppt_target(pd.DataFrame(agg, columns=cnt.columns), panel)
        vis[sec] = (feat, truth, truth_ps, list(cnt.index), centres)
        print(f"{sec}: {len(cnt)} spots, features {feat.shape}")

    secs_of = {p: sorted(s[:-4] for s in os.listdir(FEAT) if s.startswith(p) and s.endswith(".npy")
                         and not s.endswith("_patches.npy")) for p in K.HER2ST_PATIENTS}
    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    rt = []
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        ae = AE(2048, 512).to(dev)
        ae.load_state_dict(torch.load(os.path.join(a.run, "ckpt", f"{P}_ae.pt"), map_location=dev))
        mlp = Predictor(512, 512, len(panel), 0.2).to(dev)
        mlp_file = os.path.join(a.run, "ckpt", f"{P}_mlp.pt" if a.mlp == "best" else f"{P}_mlp_last.pt")
        mlp.load_state_dict(torch.load(mlp_file, map_location=dev))
        hist = pd.read_csv(os.path.join(a.run, "preds", P, "history.csv"))
        if a.mlp == "best":        # the epoch <P>_mlp.pt was last overwritten at
            epoch = int(hist.epoch[hist.val_gene_pcc.cummax().diff().fillna(1).gt(0)].iloc[-1])
        else:
            epoch = int(hist.epoch.iloc[-1])
        ae.eval(); mlp.eval()

        def run(x):
            with torch.no_grad():
                return mlp(ae.encode(torch.from_numpy(np.asarray(x, np.float32)).to(dev))).cpu().numpy()

        trainmean = np.concatenate([np.load(os.path.join(TARG, f"{s}.npy"))
                                    for q in K.HER2ST_PATIENTS if q != P for s in secs_of[q]]).mean(0)
        out_dir = os.path.join(a.out, f"fold0{k}_{P}")
        extra = dict(ae=os.path.join(a.run, "ckpt", f"{P}_ae.pt"), mlp=mlp_file,
                     epoch_rule=f"{a.mlp} (epoch {epoch})",
                     target="log10(CP10K+1), library over all detected genes",
                     train=fd["train"], test=fd["test"], visium=list(a.sections))

        if not a.skip_roundtrip:
            st = np.load(os.path.join(a.run, "preds", P, f"{P}_{epoch}.npz"), allow_pickle=True)
            for sec in secs_of[P]:
                x = np.load(os.path.join(FEAT, f"{sec}.npy"))
                sid = pd.read_csv(os.path.join(FEAT, f"{sec}_spots.csv"))["spot_id"].astype(str).tolist()
                truth = np.load(os.path.join(TARG, f"{sec}.npy"))
                pred = run(x)
                K.write_preds(out_dir, sec, pred=pred, truth=truth, spot_ids=sid, genes=panel,
                              trainmean=trainmean, model="deeppt", fold=k, extra=extra)
                m = np.array([str(s) == sec for s in st["section"]])
                sp_ = st["counts"][m]
                assert [str(s) for s in st["spot_id"][m]] == sid, f"{sec}: stored row order differs"
                rt.append(dict(model="deeppt", fold=P, section=sec, epoch=epoch,
                               max_abs_pred_diff=float(np.abs(pred - sp_).max()), max_abs_truth_diff=0.0,
                               pcc_stored=round(float(np.nanmean(per_gene_pcc(sp_, truth))), 4),
                               pcc_new=round(float(np.nanmean(per_gene_pcc(pred, truth))), 4)))
                print("  roundtrip", rt[-1])

        for sec, (feat, truth, truth_ps, sid, centres) in vis.items():
            K.write_preds(out_dir, sec, pred=run(feat), truth=truth, spot_ids=sid, genes=panel,
                          trainmean=trainmean, truth_ps=truth_ps, centre_ids=centres,
                          model="deeppt", fold=k, extra=extra)
        print(f"fold {P}: MLP = {a.mlp} epoch {epoch}, done in {time.time()-t0:.0f}s")

    if rt:
        p = K.EXT / "roundtrip.tsv"
        cur = pd.DataFrame(rt)
        df = cur
        if p.exists():
            old = pd.read_csv(p, sep="\t")
            df = pd.concat([old[~((old.model == "deeppt") & old.fold.isin(cur.fold))], cur])
        df.to_csv(p, sep="\t", index=False)
        bad = cur[(cur.pcc_new - cur.pcc_stored).abs() > a.rt_tol]
        print("ROUNDTRIP", "FAIL" if len(bad) else "PASS", f"({len(bad)} sections off by >{a.rt_tol:g})")


if __name__ == "__main__":
    main()
