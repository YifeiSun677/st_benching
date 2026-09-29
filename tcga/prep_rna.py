#!/usr/bin/env python
"""TCGA-BRCA arm, step 3 -- bulk RNA-seq -> the 833-gene panel (CPU, runs on the Mac).

For every main-cohort patient: the GDC STAR - Counts file of the primary-tumour aliquot chosen by
select_cohort.py (rna_file_id), md5-checked.

  gene naming   ENSG (version stripped; _PAR_Y rows dropped) -> the symbol her2st uses, with
                external/he_common.ensg_to_symbol (HGNC current / previous / alias, only names present
                in her2st or the panel) -- the SAME rule the He arm used.  ENSG that land on the same
                symbol are summed (as He).
  panel counts  unstranded counts of the panel genes; a panel gene no ENSG maps to is NaN (unmeasured,
                dropped from scoring, never zero)
  panel CPM     counts / sum over the measured panel genes x 1e6   (pre-registered primary target;
                matches the spot-level training targets, which are normalised within the panel)
  all genes     unstranded counts and tpm_unstranded for every gene, kept for secondary analyses

writes <out>/meta/rna/:
  rna_files.tsv                 patient, file_id, aliquot, md5 ok, gencode header
  ensg_to_symbol.tsv            the mapping table (source per ENSG)
  panel_coverage.tsv            gene, measured, n_ensg
  rna_panel_counts.tsv          patients x 833
  rna_panel_cpm.tsv             patients x 833
  rna_all_counts.tsv.gz         genes x patients (gene_id, gene_name, gene_type, then patients)
  rna_all_tpm.tsv.gz            same layout, tpm_unstranded
The downloaded STAR files are deleted afterwards.

usage: python tcga/prep_rna.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SCI = Path.home() / "Documents" / "science"
REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HER2ST_ROOT", str(SCI / "her2st" / "data"))
os.environ.setdefault("EXT_ROOT", str(SCI / "tcga_work" / "ext"))        # HGNC table cached under he/raw
sys.path.insert(0, str(REPO / "external"))
import common as K  # noqa: E402
import he_common as H  # noqa: E402

GDC_DATA = "https://api.gdc.cancer.gov/data/"


def fetch(file_id: str, md5: str, dest: Path, tries: int = 6) -> None:
    if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == md5:
        return
    for k in range(tries):
        try:
            r = requests.get(GDC_DATA + file_id, timeout=300)
            r.raise_for_status()
            if hashlib.md5(r.content).hexdigest() == md5:
                dest.write_bytes(r.content)
                return
            err = "md5 mismatch"
        except requests.RequestException as e:
            err = repr(e)
        time.sleep(15 * (k + 1))
    raise RuntimeError(f"{file_id}: download failed after {tries} tries: {err}")


def read_star(path: Path) -> tuple[pd.DataFrame, str]:
    header = path.read_text().splitlines()[0]
    df = pd.read_csv(path, sep="\t", comment="#")
    df = df[df.gene_id.str.startswith("ENSG")]                        # drops N_unmapped etc.
    df = df[~df.gene_id.str.endswith("_PAR_Y")]
    return df.set_index("gene_id"), header


def resolve_ambiguous(emap: pd.DataFrame, target: set) -> pd.DataFrame:
    """he_common.ensg_to_symbol takes the FIRST previous/alias symbol found in her2st; when that one
    already belongs to another gene the ENSG ends 'ambiguous'.  Here such an ENSG gets its other
    previous (then alias) symbol when exactly one is in her2st and unowned.
    Example: VARS1 (prev 'VARS2|VARS'); her2st has both VARS (= VARS1) and VARS2 (a different gene)."""
    h = H.hgnc_table().dropna(subset=["ensembl_gene_id"]).drop_duplicates("ensembl_gene_id")
    h = h.set_index("ensembl_gene_id")
    emap = emap.copy()
    owned = set(emap.loc[emap.source.isin(["current", "prev", "alias"]), "symbol"])
    for i in emap.index[emap.source == "ambiguous"]:
        e = emap.at[i, "ensg"]
        for src in ("prev", "alias"):
            v = h.at[e, f"{src}_symbol"]
            hits = [x for x in (v.split("|") if isinstance(v, str) else []) if x in target and x not in owned]
            if len(hits) == 1:
                emap.loc[i, ["symbol", "source"]] = [hits[0], f"{src}_resolved"]
                owned.add(hits[0])
                break
    return emap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCI / "tcga_upload"))
    ap.add_argument("--work", default=str(SCI / "tcga_work" / "rna"))
    ap.add_argument("--cohort", default=str(REPO / "results" / "tcga_cohort"))
    a = ap.parse_args()
    out = Path(a.out) / "meta" / "rna"
    work = Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    H.HE_RAW.mkdir(parents=True, exist_ok=True)

    coh = pd.read_csv(Path(a.cohort) / "cohort.csv", index_col=0)
    coh = coh[coh.role == "main"]
    man = pd.read_csv(Path(a.cohort) / "gdc_manifest_rna.txt", sep="\t").set_index("id")

    counts, tpm, files, ann = {}, {}, [], None
    for pid, row in coh.iterrows():
        fid = row.rna_file_id
        dest = work / f"{fid}.tsv"
        fetch(fid, man.loc[fid, "md5"], dest)
        df, header = read_star(dest)
        counts[pid] = df["unstranded"]
        tpm[pid] = df["tpm_unstranded"]
        if ann is None:
            ann = df[["gene_name", "gene_type"]]
        assert df.index.equals(ann.index), f"{pid}: gene order differs"
        files.append(dict(patient=pid, file_id=fid, aliquot=row.rna_aliquot, gencode=header.strip("# ")))
        print(pid, fid, flush=True)
    C = pd.DataFrame(counts)
    T = pd.DataFrame(tpm)
    pd.DataFrame(files).to_csv(out / "rna_files.tsv", sep="\t", index=False)
    pd.concat([ann, C], axis=1).to_csv(out / "rna_all_counts.tsv.gz", sep="\t", compression="gzip")
    pd.concat([ann, T], axis=1).to_csv(out / "rna_all_tpm.tsv.gz", sep="\t", compression="gzip",
                                       float_format="%.4g")

    panel = K.load_panel(REPO / "panels" / "panel_833.txt")
    ensg = [g.split(".")[0] for g in C.index]
    target = H.her2st_genes() | set(panel)
    emap = resolve_ambiguous(H.ensg_to_symbol(sorted(set(ensg)), target), target)
    emap.to_csv(out / "ensg_to_symbol.tsv", sep="\t", index=False)
    sym = pd.Series(ensg, index=C.index).map(dict(zip(emap.ensg, emap.symbol)))
    in_panel = sym.isin(panel)
    PC = C[in_panel.values].groupby(sym[in_panel].values).sum().T              # patients x measured genes
    n_ensg = sym[in_panel].value_counts()
    PC = PC.reindex(columns=panel)                                             # unmeasured -> NaN
    cov = pd.DataFrame(dict(gene=panel, measured=[int(g in n_ensg.index) for g in panel],
                            n_ensg=[int(n_ensg.get(g, 0)) for g in panel]))
    cov.to_csv(out / "panel_coverage.tsv", sep="\t", index=False)
    PC.to_csv(out / "rna_panel_counts.tsv", sep="\t")
    cpm = PC.div(PC.sum(axis=1, skipna=True), axis=0) * 1e6
    cpm.to_csv(out / "rna_panel_cpm.tsv", sep="\t", float_format="%.6g")

    summ = dict(patients=C.shape[1], genes_all=C.shape[0], panel_measured=int(cov.measured.sum()), panel_n=len(panel),
                panel_unmeasured=cov.loc[cov.measured == 0, "gene"].tolist(),
                panel_multi_ensg=cov.loc[cov.n_ensg > 1, "gene"].tolist(),
                map_source=emap.source.value_counts().to_dict(),
                panel_fraction_of_library_median=float((PC.sum(1) / C.sum(0)).median()),
                gencode=sorted({f["gencode"] for f in files}))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    for f in work.glob("*.tsv"):
        f.unlink()


if __name__ == "__main__":
    main()
