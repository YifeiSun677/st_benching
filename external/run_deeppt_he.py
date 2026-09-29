#!/usr/bin/env python
"""He Stage 4 -- DeepPT (deeppt_833_raw checkpoints) on He et al. 2020 sections.

Same encoder, weights, epoch rule and target as run_deeppt.py (the Visium driver):
  encoder   frozen ResNet50 (DeepPT_original/ResNet50_IMAGENET1K_V2.pt), 224 px crops at
            round(pixel) of the her2st-scale He JPEG, zero pad, ImageNet normalisation, fp16
            autocast -- the port's own 01_extract_features.build_encoder / encode and
            her2st_io.crop_patches.  Features cached under ext/he/features/deeppt/<SEC>.npz
  AE        <P>_ae.pt
  MLP       <P>_mlp.pt (best-val epoch, default) or <P>_mlp_last.pt (--mlp last, needs
            deeppt_resave_last.py and --run .../deeppt_833_raw_lastckpt).  USE THE SAME --run /
            --mlp AS THE VISIUM ARM AND AS STEP E.1, so both sides of the paired delta match.
  truth     02_build_targets: CP10K over the FULL count row (all detected genes), panel subset,
            log10(x + 1).  He ST-cnts keep every detected feature, so the library is complete.

The her2st held-out sections for the paired comparison come from run_deeppt.py (runbook E.1).

writes: <out>/fold0<k>_<P>/preds/<SEC>.npz   (he_common.write_preds, cohort 'he')
usage:  cd /workspace/st_benching && python external/run_deeppt_he.py [--mlp best|last] [--folds A,B]
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
import he_common as H  # noqa: E402
import run_deeppt as R  # noqa: E402  (FX encoder, io.crop_patches, AE, Predictor, deeppt_target)
from run_deeppt import AE, FX, TARG, WEIGHTS, Predictor, io  # noqa: E402

FEAT_HE = H.HE / "features" / "deeppt"


def he_features(secs, dev):
    """Encode every uncached He section once (one encoder), then return (feat, counts) per section."""
    FEAT_HE.mkdir(parents=True, exist_ok=True)
    net, out = None, {}
    for sec in secs:
        f = FEAT_HE / f"{sec}.npz"
        cnt = K.read_counts(sec, H.HE_DATA)
        if not f.exists():
            net = net or FX.build_encoder(WEIGHTS, dev)
            t0 = time.time()
            pos = K.read_spots(sec, H.HE_DATA).loc[cnt.index]
            d = H.HE_DATA / "ST-imgs" / H.metadata().loc[sec, "patient"] / sec
            img = Image.open(d / sorted(os.listdir(d))[0]).convert("RGB")
            patches = io.crop_patches(img, pos[["pixel_x", "pixel_y"]])
            img.close()
            feat = FX.encode(patches, net, dev, 256).astype(np.float32)
            np.savez(f, feat=feat, spot_id=np.array(cnt.index))
            print(f"  {sec}: {len(cnt)} spots encoded, {time.time() - t0:.0f}s", flush=True)
        z = np.load(f, allow_pickle=True)
        assert list(z["spot_id"]) == list(cnt.index), f"{sec}: cached feature rows != counts"
        out[sec] = (z["feat"], cnt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*", default=None, help="default: every exported He section")
    ap.add_argument("--subtypes", nargs="*", default=None)
    ap.add_argument("--run", default="/workspace/deeppt/results/deeppt_833_raw")
    ap.add_argument("--mlp", choices=["best", "last"], default="best")
    ap.add_argument("--out", default="/workspace/runs/he_deeppt")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    panel = K.load_panel()
    assert open(os.path.join(TARG, "genes.txt")).read().split() == panel, "targets/genes.txt != panel"

    exported = {f.name[len("counts_"):-len(".npz")] for f in H.HE_CALIB.glob("counts_*.npz")}
    secs = a.sections or [s for s in H.he_sections(subtypes=a.subtypes) if s in exported]
    missing = [s for s in secs if s not in exported]
    if missing:
        raise SystemExit(f"not exported yet: {missing} -- run export_he_like.py")
    he = {s: (feat, R.deeppt_target(cnt, panel), [str(x) for x in cnt.index])
          for s, (feat, cnt) in he_features(secs, dev).items()}

    secs_of = {p: sorted(s[:-4] for s in os.listdir(R.FEAT) if s.startswith(p) and s.endswith(".npy")
                         and not s.endswith("_patches.npy")) for p in K.HER2ST_PATIENTS}
    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        t0 = time.time()
        ae = AE(2048, 512).to(dev)
        ae.load_state_dict(torch.load(os.path.join(a.run, "ckpt", f"{P}_ae.pt"), map_location=dev))
        mlp = Predictor(512, 512, len(panel), 0.2).to(dev)
        mlp_file = os.path.join(a.run, "ckpt", f"{P}_mlp.pt" if a.mlp == "best" else f"{P}_mlp_last.pt")
        mlp.load_state_dict(torch.load(mlp_file, map_location=dev))
        hist = pd.read_csv(os.path.join(a.run, "preds", P, "history.csv"))
        if a.mlp == "best":        # the epoch <P>_mlp.pt was last overwritten at (as run_deeppt.py)
            epoch = int(hist.epoch[hist.val_gene_pcc.cummax().diff().fillna(1).gt(0)].iloc[-1])
        else:
            epoch = int(hist.epoch.iloc[-1])
        ae.eval(); mlp.eval()

        def run(x):
            with torch.no_grad():
                return mlp(ae.encode(torch.from_numpy(np.asarray(x, np.float32)).to(dev))).cpu().numpy()

        trainmean = np.concatenate([np.load(os.path.join(TARG, f"{s}.npy"))
                                    for q in K.HER2ST_PATIENTS if q != P for s in secs_of[q]]).mean(0)
        extra = dict(ae=os.path.join(a.run, "ckpt", f"{P}_ae.pt"), mlp=mlp_file, cohort="he",
                     epoch_rule=f"{a.mlp} (epoch {epoch})",
                     target="log10(CP10K+1), library over all detected genes", train=fd["train"])
        out_dir = os.path.join(out_root, f"fold0{k}_{P}")
        pcc = []
        for sec, (feat, truth, sid) in he.items():
            pred = run(feat)
            H.write_preds(out_dir, sec, pred=pred, truth=truth, spot_ids=sid, genes=panel,
                          trainmean=trainmean, model="deeppt", fold=k, extra=extra)
            pcc.append(np.nanmean(R.per_gene_pcc(pred, truth)))
        print(f"fold {P}: MLP = {a.mlp} epoch {epoch}, {len(he)} He sections, mean PCC {np.mean(pcc):.4f} "
              f"[{np.min(pcc):.4f}, {np.max(pcc):.4f}], {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
