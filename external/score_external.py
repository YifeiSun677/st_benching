#!/usr/bin/env python
"""Stage 6 -- one scorer for every model, both cohorts, both footings.

Reads   /workspace/runs/ext_<model>/fold0<k>_<P>/preds/<SEC>.npz   (common.write_preds)
        -- each fold dir holds the fold's own held-out her2st sections AND I1 I2 J1 K1,
           all predicted through the same driver, so the paired drop is on one footing.
Writes  /workspace/results/ext/per_section.csv      section x fold x footing
        /workspace/results/ext/per_patient_fold.csv  sections averaged within patient
        /workspace/results/ext/paired_delta.csv      visium patient - own her2st held-out, per fold
        /workspace/results/ext/summary.csv           mean / min / max over folds
        /workspace/results/ext/per_gene_pcc.csv.gz

Gene filter per section: in panel AND measured AND truth SD > 0.
Baseline for SSE ratio = that section's own per-gene mean (as in the main table).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common as K

MARKERS = ["ERBB2", "GRB7", "FASN", "GNAS", "ESR1", "PGR", "MKI67"]
SETS = ["all", "hvg", "svg", "marker"]


def gene_metrics(pred, truth, trainmean):
    """Column-wise metrics; pred/truth (n x g)."""
    mt = truth.mean(0)
    tc = truth - mt
    pc = pred - pred.mean(0)
    sd_t = np.sqrt((tc ** 2).sum(0))
    sd_p = np.sqrt((pc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        pcc = (tc * pc).sum(0) / (sd_t * sd_p)
        sse_base = (tc ** 2).sum(0)
        sse = ((pred - truth) ** 2).sum(0) / sse_base
        sse_tm = ((trainmean[None, :] - truth) ** 2).sum(0) / sse_base
        sdr = sd_p / sd_t
    return pcc, sse, sse_tm, sdr


def score_one(pred, truth, trainmean, genes, keep, gsets):
    g = np.array(genes)[keep]
    pcc, sse, sse_tm, sdr = gene_metrics(pred[:, keep], truth[:, keep], trainmean[keep])
    row = dict(n_spots=len(pred), n_genes=int(keep.sum()),
               pcc_mean=np.nanmean(pcc), pcc_median=np.nanmedian(pcc),
               frac_pos=np.nanmean(pcc > 0),
               sse_ratio_median=np.nanmedian(sse), frac_beat_baseline=np.nanmean(sse < 1),
               sd_ratio_median=np.nanmedian(sdr), trainmean_sse_ratio=np.nanmedian(sse_tm))
    pos = {x: i for i, x in enumerate(g)}
    for s, members in gsets.items():
        ii = [pos[x] for x in members if x in pos]
        row[f"pcc_{s}"] = np.nanmean(pcc[ii]) if ii else np.nan
        row[f"n_{s}"] = len(ii)
    for m in MARKERS:
        row[f"pcc_{m}"] = pcc[pos[m]] if m in pos else np.nan
    return row, pd.DataFrame({"gene": g, "pcc": pcc})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(K.WS / "runs"))
    ap.add_argument("--models", nargs="*", default=None, help="default: every ext_* dir")
    ap.add_argument("--out", default=str(K.WS / "results" / "ext"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gsets = {s: K.load_gene_set(s) for s in SETS}
    mdirs = sorted(glob.glob(os.path.join(a.runs, "ext_*")))
    if a.models:
        mdirs = [d for d in mdirs if os.path.basename(d)[4:] in a.models]
    rows, genes_out = [], []
    for md in mdirs:
        model = os.path.basename(md)[4:]
        for fd in sorted(glob.glob(os.path.join(md, "fold0[0-7]_[A-H]"))):
            k, held = re.search(r"fold0(\d)_([A-H])", fd).groups()
            for f in sorted(glob.glob(os.path.join(fd, "preds", "*.npz"))):
                z = np.load(f, allow_pickle=True)
                sec = str(z["section"])
                genes = list(z["genes"])
                pred, truth, tm = z["pred"].astype(np.float64), z["truth"].astype(np.float64), z["trainmean"].astype(np.float64)
                if sec not in K.VISIUM_SECTIONS and sec[0] != held:
                    raise SystemExit(f"{f}: her2st section {sec} is not the fold's held-out patient {held}")
                base = dict(model=model, fold=int(k), heldout=held, section=sec,
                            patient=K.PATIENT_OF.get(sec, sec[0]), cohort=str(z["cohort"]),
                            tier=K.TIER_OF.get(K.PATIENT_OF.get(sec, ""), 0))
                keep = z["measured"].astype(bool) & (truth.std(0) > 0)
                r, pg = score_one(pred, truth, tm, genes, keep, gsets)
                rows.append({**base, "footing": "native", **r})
                genes_out.append(pg.assign(**base, footing="native"))
                if "truth_ps" in z.files:
                    tps = z["truth_ps"].astype(np.float64)
                    pps = pred[z["centre_idx"]]
                    keep_ps = z["measured"].astype(bool) & (tps.std(0) > 0)
                    r, pg = score_one(pps, tps, tm, genes, keep_ps, gsets)
                    rows.append({**base, "footing": "pseudo7", **r})
                    genes_out.append(pg.assign(**base, footing="pseudo7"))
    if not rows:
        raise SystemExit("no prediction files found")
    sec_df = pd.DataFrame(rows)
    sec_df.to_csv(os.path.join(a.out, "per_section.csv"), index=False)
    pd.concat(genes_out).to_csv(os.path.join(a.out, "per_gene_pcc.csv.gz"), index=False)

    num = [c for c in sec_df.columns if c not in
           ("model", "fold", "heldout", "section", "patient", "cohort", "tier", "footing")]
    pat = (sec_df.groupby(["model", "fold", "heldout", "patient", "cohort", "tier", "footing"])[num]
           .mean().reset_index())
    pat.to_csv(os.path.join(a.out, "per_patient_fold.csv"), index=False)

    # paired drop: native footing only (her2st has no pseudo-spot footing)
    nat = pat[pat.footing == "native"]
    own = nat[nat.cohort == "her2st"].set_index(["model", "fold"])
    vis = nat[nat.cohort == "visium"]
    d = []
    for _, v in vis.iterrows():
        o = own.loc[(v.model, v.fold)]
        d.append({"model": v.model, "fold": v.fold, "heldout": v.heldout, "patient": v.patient,
                  "tier": v.tier, **{f"delta_{c}": v[c] - o[c] for c in num if not c.startswith("n_")}})
    delta = pd.DataFrame(d)
    delta.to_csv(os.path.join(a.out, "paired_delta.csv"), index=False)

    key = ["pcc_mean", "frac_beat_baseline", "sse_ratio_median", "sd_ratio_median",
           "trainmean_sse_ratio", "pcc_ERBB2", "pcc_GRB7"]
    summ = (pat.assign(patient=np.where(pat.cohort == "her2st", "her2st_heldout", pat.patient))
            .groupby(["model", "patient", "footing"])[key].agg(["mean", "min", "max"]))
    summ.columns = [f"{a_}_{b}" for a_, b in summ.columns]
    dsum = delta.groupby(["model", "patient"])[["delta_pcc_mean", "delta_frac_beat_baseline"]].agg(["mean", "min", "max"])
    dsum.columns = [f"{a_}_{b}" for a_, b in dsum.columns]
    dsum = dsum.reset_index().assign(footing="native")      # the paired drop exists on the native footing only
    summ = summ.reset_index().merge(dsum, on=["model", "patient", "footing"], how="left")
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summ[["model", "patient", "footing", "pcc_mean_mean", "pcc_mean_min", "pcc_mean_max",
                    "frac_beat_baseline_mean", "delta_pcc_mean_mean"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
