"""external/tcga_common.py -- shared helpers for the TCGA-BRCA arm (tcga/RUNBOOK_TCGA.md).

Layout on the pod (after the upload, runbook B.1):
  /workspace/ext/tcga/her2st_like/data/   ST-imgs/<patient>/<SEC>/HE_<SEC>.jpg, ST-spotfiles, ST-cnts
                                          (ST-cnts are PLACEHOLDERS -- never a target)
  /workspace/ext/tcga/meta/               params.json, windows.tsv, slides.tsv, cohort.csv, rna/

TCGA has no spot truth, so TCGA drivers do not write pred/truth pairs.  Per model x fold x section
they write AGGREGATES (all spots, full precision):

  <out>/fold0<k>_<P>/agg/<SEC>.npz
      sum_lin   (833,) float64  sum over spots of the prediction on the LINEAR scale (the model's
                                own inverse transform: e.g. exp for ST-Net's log((1+c)/(n+Z)))
      sum_log   (833,) float64  sum over spots of the raw prediction (model's own space)
      n_spots   int
      genes, section, patient, kind (TS/BS/DX), token, model, fold, inverse (text)
  <out>/fold0<k>_<P>/spots/<SEC>.npz      only for --spots-for patients: pred (float16, raw space),
                                          spot_id -- for maps, not for scoring

Slide- and patient-level pseudo-bulk are then exact:  sum(sum_lin) / sum(n_spots)  (tcga/pseudobulk.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import common as K

TCGA = Path(os.environ.get("TCGA_ROOT", K.EXT / "tcga"))
TCGA_DATA = TCGA / "her2st_like" / "data"
TCGA_META = TCGA / "meta"


def windows() -> pd.DataFrame:
    """meta/windows.tsv (one row per section) + stratum; only sections whose files are on disk
    (the upload is extracted batch by batch)."""
    w = pd.read_csv(TCGA_META / "windows.tsv", sep="\t")
    on_disk = w.section.map(lambda s: (TCGA_DATA / "ST-spotfiles" / f"{s}_selection.tsv").exists())
    return w[on_disk].reset_index(drop=True)


def select_sections(sections=None, patients=None, kinds=None, batch=None, limit=0) -> pd.DataFrame:
    w = windows()
    if sections:
        w = w[w.section.isin(sections)]
    if patients:
        w = w[w.patient.isin(patients)]
    if kinds:
        w = w[w.kind.isin(kinds)]
    if batch:
        w = w[w.batch.isin(batch)]
    if limit:
        w = w.head(limit)
    return w.reset_index(drop=True)


def image_path(sec: str) -> Path:
    patient = sec.split("_")[0]
    return TCGA_DATA / "ST-imgs" / patient / sec / f"HE_{sec}.jpg"


def read_spots(sec: str) -> pd.DataFrame:
    return K.read_spots(sec, TCGA_DATA)


def default_spot_patients() -> list[str]:
    """Rank-1 main patient of each stratum: their full spot matrices are kept for maps."""
    c = pd.read_csv(TCGA_META / "cohort.csv", index_col=0)
    c = c[c.role == "main"].sort_values(["stratum", "rank"])
    return list(c.groupby("stratum").head(1).index)


def agg_path(out_dir: Path | str, sec: str) -> Path:
    return Path(out_dir) / "agg" / f"{sec}.npz"


def write_agg(out_dir: Path | str, row, *, pred_raw: np.ndarray, pred_lin: np.ndarray, spot_ids, genes,
              model: str, fold: int, inverse: str, save_spots: bool = False, extra: dict | None = None):
    pred_raw = np.asarray(pred_raw, np.float64)
    pred_lin = np.asarray(pred_lin, np.float64)
    assert pred_raw.shape == pred_lin.shape == (len(spot_ids), len(genes)), (pred_raw.shape, len(spot_ids))
    assert np.isfinite(pred_raw).all() and np.isfinite(pred_lin).all(), f"{row.section}: non-finite predictions"
    p = agg_path(out_dir, row.section)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, sum_lin=pred_lin.sum(0), sum_log=pred_raw.sum(0), n_spots=len(spot_ids),
             sd_raw=pred_raw.std(0), genes=np.array(genes), section=row.section, patient=row.patient,
             kind=row.kind, token=row.token, model=model, fold=int(fold), inverse=inverse)
    if save_spots:
        s = Path(out_dir) / "spots" / f"{row.section}.npz"
        s.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(s, pred=pred_raw.astype(np.float16), spot_id=np.array(spot_ids), genes=np.array(genes))
    if extra:
        (Path(out_dir) / "run.json").write_text(json.dumps(extra, indent=2, default=str))
