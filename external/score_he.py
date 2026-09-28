#!/usr/bin/env python
"""He Stage 5 -- score her2st-trained models on He et al. 2020, paired with their own her2st held-out.

Reads   /workspace/runs/he_<model>/fold0<k>_<P>/preds/<SEC>.npz
          He sections        (run_stnet_he.py, cohort 'he')
          her2st held-out    (run_stnet.py --sections --out <same dir>, cohort 'her2st')
Writes  /workspace/results/he/per_section.csv       section x fold (+ fold 'ens' = mean of the 8 folds)
        /workspace/results/he/per_patient_fold.csv  sections averaged within patient
        /workspace/results/he/paired_delta.csv      He patient - the fold's own her2st held-out patient
        /workspace/results/he/summary.csv           by subtype group, mean / min / max over folds
        /workspace/results/he/per_gene_pcc.csv.gz

Gene filter per section: in panel AND measured AND truth SD > 0 (same as score_external.py);
metrics are score_external.score_one, so numbers are on the same footing as the Visium arm.
Subtype groups: each subtype, HER2+ (luminal + non-luminal = same disease as her2st), He_all.
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
import he_common as H
from score_external import SETS, score_one

KEY = ["pcc_mean", "pcc_hvg", "pcc_svg", "pcc_marker", "frac_beat_baseline", "sse_ratio_median",
       "sd_ratio_median", "trainmean_sse_ratio", "pcc_ERBB2", "pcc_GRB7", "pcc_ESR1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(K.WS / "runs"))
    ap.add_argument("--models", nargs="*", default=None, help="default: every he_* dir")
    ap.add_argument("--out", default=str(K.WS / "results" / "he"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gsets = {s: K.load_gene_set(s) for s in SETS}
    md = H.metadata()
    mdirs = sorted(glob.glob(os.path.join(a.runs, "he_*")))
    if a.models:
        mdirs = [d for d in mdirs if os.path.basename(d)[3:] in a.models]
    rows, genes_out = [], []

    def add(base, pred, truth, tm, genes, measured):
        keep = measured & (truth.std(0) > 0)
        r, pg = score_one(pred, truth, tm, genes, keep, gsets)
        rows.append({**base, **r})
        genes_out.append(pg.assign(**base))

    for mdir in mdirs:
        model = os.path.basename(mdir)[3:]
        ens = {}
        for fd in sorted(glob.glob(os.path.join(mdir, "fold0[0-7]_[A-H]"))):
            k, held = re.search(r"fold0(\d)_([A-H])", fd).groups()
            for f in sorted(glob.glob(os.path.join(fd, "preds", "*.npz"))):
                z = np.load(f, allow_pickle=True)
                sec, cohort = str(z["section"]), str(z["cohort"])
                genes = list(z["genes"])
                pred, truth, tm = (z[x].astype(np.float64) for x in ("pred", "truth", "trainmean"))
                if cohort == "her2st":
                    if sec[0] != held:
                        raise SystemExit(f"{f}: her2st section {sec} is not the fold's held-out patient {held}")
                    patient, subtype = sec[0], "her2st_heldout"
                else:
                    patient, subtype = md.loc[sec, "patient"], md.loc[sec, "type"]
                    e = ens.setdefault(sec, dict(pred=0, n=0, truth=truth, tm=0, genes=genes,
                                                 measured=z["measured"].astype(bool), patient=patient,
                                                 subtype=subtype))
                    e["pred"] = e["pred"] + pred; e["tm"] = e["tm"] + tm; e["n"] += 1
                add(dict(model=model, fold=k, heldout=held, section=sec, patient=patient, subtype=subtype,
                         cohort=cohort), pred, truth, tm, genes, z["measured"].astype(bool))
        for sec, e in ens.items():
            add(dict(model=model, fold="ens", heldout=f"ens{e['n']}", section=sec, patient=e["patient"],
                     subtype=e["subtype"], cohort="he"), e["pred"] / e["n"], e["truth"], e["tm"] / e["n"],
                e["genes"], e["measured"])
    if not rows:
        raise SystemExit("no prediction files found")

    sec_df = pd.DataFrame(rows)
    sec_df.to_csv(os.path.join(a.out, "per_section.csv"), index=False)
    pd.concat(genes_out).to_csv(os.path.join(a.out, "per_gene_pcc.csv.gz"), index=False)
    num = [c for c in sec_df.columns if c not in
           ("model", "fold", "heldout", "section", "patient", "subtype", "cohort")]
    pat = (sec_df.groupby(["model", "fold", "heldout", "patient", "subtype", "cohort"])[num]
           .mean().reset_index())
    pat.to_csv(os.path.join(a.out, "per_patient_fold.csv"), index=False)

    # paired drop: each He patient vs the SAME fold's her2st held-out patient
    per = pat[pat.fold != "ens"]
    own = per[per.cohort == "her2st"].set_index(["model", "fold"])
    d = []
    for _, v in per[per.cohort == "he"].iterrows():
        if (v.model, v.fold) not in own.index:
            continue
        o = own.loc[(v.model, v.fold)]
        d.append({"model": v.model, "fold": v.fold, "heldout": v.heldout, "patient": v.patient,
                  "subtype": v.subtype, **{f"delta_{c}": v[c] - o[c] for c in num if not c.startswith("n_")}})
    delta = pd.DataFrame(d)
    delta.to_csv(os.path.join(a.out, "paired_delta.csv"), index=False)
    if delta.empty:
        print("NOTE: no her2st held-out preds in the run dir -> no paired delta (runbook step 4.1)")

    # summary: patients averaged within fold, then mean/min/max over the 8 folds
    groups = {s: [s] for s in sorted(pat.subtype.unique())}
    groups.update({"HER2+": list(H.HER2_SUBTYPES), "He_all": sorted(md["type"].unique())})
    out = []
    for g, members in groups.items():
        sub = pat[pat.subtype.isin(members)]
        if sub.empty:
            continue
        for (model, is_ens), s in sub.groupby(["model", sub.fold == "ens"]):
            byfold = s.groupby("fold")[KEY].mean()
            r = dict(model=model, group=g, folds="ens" if is_ens else "8fold",
                     n_patients=s.patient.nunique(), n_folds=len(byfold))
            for c in KEY:
                r[f"{c}_mean"], r[f"{c}_min"], r[f"{c}_max"] = byfold[c].mean(), byfold[c].min(), byfold[c].max()
            if not is_ens and not delta.empty and g != "her2st_heldout":
                dd = delta[(delta.model == model) & delta.subtype.isin(members)].groupby("fold").delta_pcc_mean.mean()
                r["delta_pcc_mean_mean"], r["delta_pcc_mean_min"], r["delta_pcc_mean_max"] = dd.mean(), dd.min(), dd.max()
            out.append(r)
    summ = pd.DataFrame(out)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    cols = [c for c in ["model", "group", "folds", "n_patients", "pcc_mean_mean", "pcc_mean_min", "pcc_mean_max",
                        "frac_beat_baseline_mean", "pcc_ERBB2_mean", "delta_pcc_mean_mean"] if c in summ]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summ[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
