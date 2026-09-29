#!/usr/bin/env python
"""TCGA-BRCA arm, step 1 -- cohort selection (metadata only, runs locally, no GPU).

Design (fixed before any prediction is run):
  strata   HER2pos  clinical HER2+ (any PAM50)                 40
           LumA     clinical HER2- and PAM50 LumA               20
           LumB     clinical HER2- and PAM50 LumB               20
           Basal    clinical HER2- and PAM50 Basal              20
  eligible female; no neoadjuvant therapy; primary-tumour RNA-seq (STAR counts);
           >=1 frozen slide (TS/BS) from the SAME sample as the RNA; >=1 DX slide;
           determinable clinical HER2 status.
  order    per stratum, eligible patients are sorted by ID and permuted with a fixed
           seed.  Ranks 1..n = main, the rest = reserve.
           A main patient that fails slide QC is replaced by the next reserve patient
           of the same stratum, strictly in rank order.

Clinical HER2 (ASCO/CAP logic on the brca_tcga fields):
  FISH Positive                                   -> pos
  FISH Negative and IHC not Positive              -> neg
  FISH Negative and IHC Positive                  -> discordant (excluded)
  no usable FISH: IHC Positive -> pos, IHC Negative -> neg, else undetermined (excluded)

Sources: GDC files API (slides, RNA-seq), cBioPortal brca_tcga (HER2 IHC/FISH, sex,
neoadjuvant) and brca_tcga_pan_can_atlas_2018 (PAM50 SUBTYPE).  Raw API responses
are saved under <out>/raw/ so the selection can be reproduced after the APIs change.

usage: python tcga/select_cohort.py [--out results/tcga_cohort] [--seed 20260929] [--offline]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GDC = "https://api.gdc.cancer.gov/files"
CBIO = "https://www.cbioportal.org/api/studies/{study}/clinical-data"
STRATA = {"HER2pos": 40, "LumA": 20, "LumB": 20, "Basal": 20}
PAM50_OF = {"BRCA_LumA": "LumA", "BRCA_LumB": "LumB", "BRCA_Basal": "Basal",
            "BRCA_Her2": "Her2", "BRCA_Normal": "Normal"}
SLIDE_RE = re.compile(r"^(TCGA-\w\w-\w{4})-(\d\d)([A-Z])-(\d\d)-([A-Z]{2})([0-9A-Z]+)\.")


# ------------------------------------------------------------------ fetch ----
def gdc_query(data_type: str, extra: list, fields: list[str]) -> list[dict]:
    flt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "data_type", "value": [data_type]}}, *extra]}
    r = requests.post(GDC, json=dict(filters=flt, fields=",".join(fields), format="JSON", size=20000),
                      timeout=300)
    r.raise_for_status()
    d = r.json()["data"]
    assert d["pagination"]["total"] == len(d["hits"]), "GDC result truncated; raise size"
    return d["hits"]


def cbio_patients(study: str) -> list[dict]:
    r = requests.get(CBIO.format(study=study), timeout=300,
                     params=dict(clinicalDataType="PATIENT", projection="SUMMARY", pageSize=10**7))
    r.raise_for_status()
    return r.json()


def fetch(raw: Path, offline: bool) -> dict:
    files = {"slides": raw / "gdc_slides.json.gz", "rna": raw / "gdc_rna.json.gz",
             "cbio_brca": raw / "cbio_brca_tcga_patient.json.gz",
             "cbio_pancan": raw / "cbio_pancan_patient.json.gz"}
    if not offline:
        raw.mkdir(parents=True, exist_ok=True)
        base = ["file_id", "file_name", "file_size", "md5sum", "state", "cases.submitter_id",
                "cases.samples.submitter_id", "cases.samples.sample_type"]
        got = {
            "slides": gdc_query("Slide Image", [], base + ["experimental_strategy"]),
            "rna": gdc_query("Gene Expression Quantification",
                             [{"op": "in", "content": {"field": "analysis.workflow_type",
                                                       "value": ["STAR - Counts"]}}],
                             base + ["cases.samples.portions.analytes.aliquots.submitter_id"]),
            "cbio_brca": cbio_patients("brca_tcga"),
            "cbio_pancan": cbio_patients("brca_tcga_pan_can_atlas_2018"),
        }
        for k, v in got.items():
            with gzip.open(files[k], "wt") as fh:
                json.dump(v, fh)
    return {k: json.load(gzip.open(p, "rt")) for k, p in files.items()}


# ----------------------------------------------------------------- tables ----
def slide_table(hits: list[dict]) -> pd.DataFrame:
    rows = []
    for h in hits:
        m = SLIDE_RE.match(h["file_name"])
        if not m:
            raise ValueError(f"unparsed slide name {h['file_name']}")
        pat, st, vial, _, kind, suf = m.groups()
        rows.append(dict(patient=pat, sample=f"{pat}-{st}{vial}", sample_type_code=st, kind=kind, token=kind + suf,
                         file_id=h["file_id"], file_name=h["file_name"], file_size=h["file_size"],
                         md5sum=h["md5sum"], state=h["state"]))
    return pd.DataFrame(rows)


def rna_table(hits: list[dict]) -> pd.DataFrame:
    rows = []
    for h in hits:
        (case,) = h["cases"]
        (smp,) = case["samples"]
        aliq = [a["submitter_id"] for p in smp.get("portions", []) for an in p.get("analytes", [])
                for a in an.get("aliquots", [])]
        rows.append(dict(patient=case["submitter_id"], sample=smp["submitter_id"],
                         sample_type=smp["sample_type"], aliquot=";".join(sorted(aliq)),
                         file_id=h["file_id"], file_name=h["file_name"], file_size=h["file_size"],
                         md5sum=h["md5sum"], state=h["state"]))
    return pd.DataFrame(rows)


def pivot_cbio(rows: list[dict]) -> pd.DataFrame:
    return (pd.DataFrame(rows).pivot_table(index="patientId", columns="clinicalAttributeId",
                                           values="value", aggfunc="first"))


def her2_status(ihc, fish) -> str:
    if fish == "Positive":
        return "pos"
    if fish == "Negative":
        return "discordant" if ihc == "Positive" else "neg"
    return {"Positive": "pos", "Negative": "neg"}.get(ihc, "undetermined")


# --------------------------------------------------------------- selection ----
def build(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sl, rn = slide_table(data["slides"]), rna_table(data["rna"])
    # GDC occasionally holds two scans of the same physical slide (same sample and token, different
    # UUID and scanner; e.g. TCGA-A8-A06U-01A-01-TS1).  Keep one per slide: the lowest file_id.
    sl = sl.sort_values("file_id").drop_duplicates(["sample", "token"], keep="first")
    cb, pc = pivot_cbio(data["cbio_brca"]), pivot_cbio(data["cbio_pancan"])

    # one RNA file per patient: primary tumour, vial A preferred, then lowest aliquot barcode
    rn = rn[rn.sample_type == "Primary Tumor"].copy()
    rn["vial"] = rn["sample"].str[-1]
    rn = rn.sort_values(["patient", "vial", "aliquot", "file_id"])
    n_rna = rn.groupby("patient").size().rename("n_rna_files")
    rna1 = rn.groupby("patient").head(1).set_index("patient")

    pts = sorted(set(cb.index) | set(pc.index) | set(rna1.index) | set(sl.patient))
    P = pd.DataFrame(index=pd.Index(pts, name="patient"))
    P["sex"] = pc["SEX"].reindex(pts).fillna(cb["SEX"].reindex(pts))
    neo = pd.concat([cb["HISTORY_NEOADJUVANT_TRTYN"].reindex(pts),
                     pc["HISTORY_NEOADJUVANT_TRTYN"].reindex(pts)], axis=1)
    P["neoadjuvant"] = np.where((neo == "Yes").any(axis=1), "Yes",
                                np.where((neo == "No").any(axis=1), "No", "unknown"))
    P["ihc_her2"] = cb["IHC_HER2"].reindex(pts)
    P["fish_her2"] = cb["HER2_FISH_STATUS"].reindex(pts)
    P["her2"] = [her2_status(i, f) for i, f in zip(P.ihc_her2, P.fish_her2)]
    P["pam50"] = pc["SUBTYPE"].reindex(pts).map(PAM50_OF)
    P["rna_sample"] = rna1["sample"].reindex(pts)
    P["rna_aliquot"] = rna1["aliquot"].reindex(pts)
    P["rna_file_id"] = rna1["file_id"].reindex(pts)
    P["n_rna_files"] = n_rna.reindex(pts).fillna(0).astype(int)

    frozen = sl[sl.kind.isin(["TS", "BS"])].merge(P[["rna_sample"]], left_on="patient", right_index=True)
    frozen = frozen[frozen["sample"] == frozen["rna_sample"]]            # same sample as the RNA
    dx = sl[(sl.kind == "DX") & (sl.sample_type_code == "01")]
    P["n_TS"] = frozen[frozen.kind == "TS"].groupby("patient").size().reindex(pts).fillna(0).astype(int)
    P["n_BS"] = frozen[frozen.kind == "BS"].groupby("patient").size().reindex(pts).fillna(0).astype(int)
    P["n_DX"] = dx.groupby("patient").size().reindex(pts).fillna(0).astype(int)

    P["stratum"] = np.select(
        [P.her2 == "pos", (P.her2 == "neg") & P.pam50.isin(["LumA", "LumB", "Basal"])],
        ["HER2pos", P.pam50.fillna("")], default="")

    # exclusion reasons, first failing rule wins (CONSORT-style flow)
    rules = [
        ("not_female", P.sex != "Female"),
        ("neoadjuvant_yes_or_unknown", P.neoadjuvant != "No"),
        ("no_primary_rna", P.rna_sample.isna()),
        ("no_frozen_slide_same_sample", (P.n_TS + P.n_BS) == 0),
        ("no_dx_slide", P.n_DX == 0),
        ("her2_discordant", P.her2 == "discordant"),
        ("her2_undetermined", P.her2 == "undetermined"),
        ("her2neg_pam50_not_LumA_LumB_Basal", P.stratum == ""),
    ]
    P["exclusion"] = ""
    for name, mask in rules:
        P.loc[(P.exclusion == "") & mask.values, "exclusion"] = name
    return P, frozen, dx


def assign(P: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    P = P.copy()
    P["rank"], P["role"] = np.nan, ""
    for s, n_main in STRATA.items():         # fixed stratum order -> reproducible draws
        ids = sorted(P.index[(P.exclusion == "") & (P.stratum == s)])
        order = [ids[i] for i in rng.permutation(len(ids))]
        for r, pid in enumerate(order, 1):
            P.loc[pid, "rank"] = r
            P.loc[pid, "role"] = "main" if r <= n_main else "reserve"
    return P


# ------------------------------------------------------------------ write ----
def gdc_manifest(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(dict(id=df.file_id, filename=df.file_name, md5=df.md5sum,
                             size=df.file_size, state=df.state))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tcga_cohort")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--offline", action="store_true", help="reuse <out>/raw instead of calling the APIs")
    a = ap.parse_args()
    out = Path(a.out)
    data = fetch(out / "raw", a.offline)
    P, frozen, dx = build(data)
    P = assign(P, a.seed)

    P.to_csv(out / "eligibility.csv")
    sel = P[P.role == "main"]
    cohort = P[P.role != ""].sort_values(["stratum", "rank"])
    cohort.to_csv(out / "cohort.csv")

    slides = pd.concat([frozen.drop(columns="rna_sample"), dx])
    slides = slides[slides.patient.isin(cohort.index)].merge(
        cohort[["stratum", "rank", "role"]], left_on="patient", right_index=True)
    slides = slides.sort_values(["stratum", "rank", "kind", "file_name"])
    slides.to_csv(out / "slides.csv", index=False)
    s_sel = slides[slides.role == "main"]
    gdc_manifest(s_sel).to_csv(out / "gdc_manifest_slides.txt", sep="\t", index=False)
    rn = rna_table(data["rna"]).set_index("file_id").loc[sel.rna_file_id]
    gdc_manifest(rn.reset_index()).to_csv(out / "gdc_manifest_rna.txt", sep="\t", index=False)

    elig = P[P.exclusion == ""]
    summary = dict(
        built=dt.datetime.now().isoformat(timespec="seconds"), seed=a.seed, strata=STRATA,
        n_patients_any_source=len(P),
        exclusion_flow=P.exclusion.replace("", "eligible").value_counts().to_dict(),
        eligible_by_stratum=elig.stratum.value_counts().to_dict(),
        her2pos_pam50=elig[elig.stratum == "HER2pos"].pam50.fillna("NA").value_counts().to_dict(),
        selected=sel.stratum.value_counts().to_dict(),
        shortfall={s: max(0, n - int((elig.stratum == s).sum())) for s, n in STRATA.items()},
        selected_slides={k: int(v) for k, v in s_sel.kind.value_counts().items()},
        selected_slide_gb=round(s_sel.file_size.sum() / 1e9, 1),
        main_with_both_TS_and_BS=int(((sel.n_TS > 0) & (sel.n_BS > 0)).sum()),
        patients_with_multiple_rna_files=int((sel.n_rna_files > 1).sum()),
        sources=dict(gdc=GDC, cbio_her2="brca_tcga", cbio_pam50="brca_tcga_pan_can_atlas_2018"),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
