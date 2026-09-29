#!/usr/bin/env python
"""TCGA-BRCA arm, step 4 -- pack the Mac-side outputs for upload to the pod.

  batch_NN.tar   her2st-layout sections (ST-imgs / ST-spotfiles / ST-cnts) for a set of whole
                 patients (all of a patient's slides in one batch), each <= --max-gb of JPEG.
                 Uncompressed tar (the JPEGs do not compress).  Extract on the pod with
                   tar -xf batch_NN.tar -C /workspace/ext/tcga/her2st_like
                 which gives .../her2st_like/data/ST-imgs/..., as the He arm's her2st_like/data.
  meta.tar.gz    meta/ (params.json, slides.tsv, windows.tsv, batches.tsv, rna/, cohort files)
                 and qc/ (one thumbnail per slide)

Each tar is listed back and checked against the files it should hold before the loose data/ is
deleted (--keep-loose to skip the deletion).  meta/ and qc/ are small and stay loose for browsing.

usage: python tcga/pack_upload.py [--max-gb 8]
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

import pandas as pd

SCI = Path.home() / "Documents" / "science"
REPO = Path(__file__).resolve().parents[1]


def section_files(data: Path, row) -> list[Path]:
    sec = row.section
    return [data / "ST-imgs" / row.patient / sec / f"HE_{sec}.jpg",
            data / "ST-spotfiles" / f"{sec}_selection.tsv",
            data / "ST-cnts" / f"{sec}.tsv.gz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCI / "tcga_upload"))
    ap.add_argument("--cohort", default=str(REPO / "results" / "tcga_cohort"))
    ap.add_argument("--max-gb", type=float, default=8.0)
    ap.add_argument("--keep-loose", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    data, meta = out / "data", out / "meta"

    sl = pd.read_csv(meta / "slides.tsv", sep="\t")
    win = pd.read_csv(meta / "windows.tsv", sep="\t")
    bad = sl[sl.status != "ok"]
    if len(bad):
        print("slides not ok (not packed):\n" + bad[["patient", "token", "status"]].to_string(index=False))

    # whole patients into batches, biggest first, first-fit
    per_pat = win.groupby("patient").jpg_mb.sum().sort_values(ascending=False)
    cap = a.max_gb * 1e3
    batches: list[list[str]] = []
    sizes: list[float] = []
    for pid, mb in per_pat.items():
        for i, s in enumerate(sizes):
            if s + mb <= cap:
                batches[i].append(pid)
                sizes[i] += mb
                break
        else:
            batches.append([pid])
            sizes.append(mb)
    bmap = {p: i + 1 for i, ps in enumerate(batches) for p in ps}
    win["batch"] = win.patient.map(bmap)
    win.to_csv(meta / "windows.tsv", sep="\t", index=False)
    summary = win.groupby("batch").agg(patients=("patient", "nunique"), sections=("section", "size"),
                                       spots=("n_spots", "sum"), jpg_gb=("jpg_mb", lambda x: round(x.sum() / 1e3, 2)))
    summary.to_csv(meta / "batches.tsv", sep="\t")
    print(summary.to_string())

    for src, dst in (("cohort.csv", "cohort.csv"), ("slides.csv", "cohort_slides.csv"),
                     ("summary.json", "cohort_summary.json")):
        shutil.copy(Path(a.cohort) / src, meta / dst)

    all_files = []
    for b, g in win.groupby("batch"):
        tar = out / f"batch_{b:02d}.tar"
        files = [p for r in g.itertuples() for p in section_files(data, r)]
        missing = [p for p in files if not p.exists()]
        assert not missing, f"batch {b}: missing {missing[:3]}"
        with tarfile.open(tar, "w") as t:
            for p in files:
                t.add(p, arcname=str(p.relative_to(out)))
        with tarfile.open(tar) as t:
            names = {m.name: m.size for m in t.getmembers()}
        for p in files:
            assert names.get(str(p.relative_to(out))) == p.stat().st_size, f"{tar}: bad member {p}"
        all_files += files
        print(f"{tar.name}: {len(files)} files, {tar.stat().st_size / 1e9:.2f} GB, verified", flush=True)

    mt = out / "meta.tar.gz"
    with tarfile.open(mt, "w:gz") as t:
        t.add(meta, arcname="meta")
        t.add(out / "qc", arcname="qc")
    print(f"{mt.name}: {mt.stat().st_size / 1e6:.1f} MB")

    if not a.keep_loose:
        shutil.rmtree(data)
        print("loose data/ removed (inside batch_*.tar); meta/ and qc/ kept for browsing")


if __name__ == "__main__":
    main()
