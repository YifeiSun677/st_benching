"""external/he_common.py -- shared helpers for the her2st -> He et al. 2020 transfer arm.

He et al. 2020 (ST-Net paper) cohort: Mendeley 10.17632/29ntw7sh4r, 23 patients, 68 sections,
legacy ST (100 um spots, 200 um pitch -- the same platform as her2st), subtypes HER2_luminal,
HER2_non_luminal, Luminal_A, Luminal_B, TNBC.  No He patient is in any her2st training fold, so
every fold model predicts every He section.

Raw files per section (names come from raw/metadata.csv; the BT/BC prefixes are NOT consistent):
  HE_<BT|BC><id>_<rep>.jpg          full-res H&E, ~9300 x 9900 px, ~291 px per 200 um
  spots_<BT|BC><id>_<rep>.csv.gz    index 'XxY' (array col x row), X, Y = pixel column, row
                                    (under-tissue spots only)
  BC<id>_<rep>_stdata.tsv.gz        spots x ENSG counts, index 'XxY', ALL array spots (~1000),
                                    genes with 0 counts in the section are dropped (her2st-like)
  BC<id>_<rep>_Coords.tsv.gz        tumour annotation, CR line endings, 4 header names / 5 cols:
                                    <rep>_x_y  x  y  L<n>  tumor|non

Layout on the pod (kept SEPARATE from her2st and from the Visium arm, same reasons as common.py):
  /workspace/ext/he/raw/                  Mendeley downloads + metadata.csv + MANIFEST.tsv
  /workspace/ext/he/her2st_like/data/     ST-imgs/<patient>/<SEC>/HE_<SEC>.jpg (resampled),
                                          ST-spotfiles/<SEC>_selection.tsv, ST-cnts/<SEC>.tsv.gz
  /workspace/ext/he/calib/                he_scale.tsv, counts_<SEC>.npz, panel_coverage.tsv, qc/
Section id <SEC> = metadata patient + '_' + replicate, e.g. BC23287_C1.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

import common as K

HE = Path(os.environ.get("HE_ROOT", K.EXT / "he"))
HE_RAW = HE / "raw"
HE_DATA = HE / "her2st_like" / "data"
HE_CALIB = HE / "calib"
MENDELEY_ID = "29ntw7sh4r"
HER2_SUBTYPES = ("HER2_luminal", "HER2_non_luminal")


# ---------------------------------------------------------------- metadata ----
def metadata() -> pd.DataFrame:
    df = pd.read_csv(HE_RAW / "metadata.csv")
    df["section"] = df["patient"] + "_" + df["replicate"]
    return df.set_index("section", drop=False)


def he_sections(subtypes=None, patients=None) -> list[str]:
    md = metadata()
    if subtypes:
        md = md[md["type"].isin(subtypes)]
    if patients:
        md = md[md["patient"].isin(patients)]
    return list(md.index)


def raw_paths(sec: str) -> dict:
    row = metadata().loc[sec]
    return {"image": HE_RAW / row["histology_image"], "spots": HE_RAW / row["spot_coordinates"],
            "counts": HE_RAW / row["count_matrix"], "tumor": HE_RAW / row["tumor_annotation"],
            "patient": row["patient"], "subtype": row["type"]}


# ------------------------------------------------------------- raw readers ----
def _xy_from_ids(ids) -> tuple[np.ndarray, np.ndarray]:
    xy = np.array([[float(v) for v in str(i).split("x")] for i in ids])
    if not np.allclose(xy, np.round(xy)):
        raise ValueError("non-integer spot ids")
    return np.round(xy[:, 0]).astype(int), np.round(xy[:, 1]).astype(int)


def read_raw_spots(path: Path) -> pd.DataFrame:
    """-> index 'XxY', columns x, y (array units, 200 um), pixel_x (column), pixel_y (row)."""
    df = pd.read_csv(path, index_col=0)
    x, y = _xy_from_ids(df.index)
    out = pd.DataFrame({"x": x, "y": y, "pixel_x": df["X"].values, "pixel_y": df["Y"].values})
    out.index = [f"{a}x{b}" for a, b in zip(x, y)]
    assert out.index.is_unique, f"{path}: duplicate spot ids"
    return out


def read_raw_counts(path: Path) -> pd.DataFrame:
    """-> spots x ENSG raw counts (int), index 'XxY'."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    x, y = _xy_from_ids(df.index)
    df.index = [f"{a}x{b}" for a, b in zip(x, y)]
    v = df.values
    if not np.allclose(v, np.round(v)):
        raise ValueError(f"{path}: non-integer counts")
    return df.round().astype(np.int32)


def read_tumor(path: Path) -> dict:
    """Coords file (CR line endings, misaligned header) -> {'XxY': 'tumor' | 'non'}."""
    with gzip.open(path, "rt", newline="") as f:
        lines = [ln for ln in re.split(r"\r\n|\r|\n", f.read()) if ln.strip()]
    out = {}
    for ln in lines[1:]:
        p = ln.split("\t")
        out[f"{int(round(float(p[1])))}x{int(round(float(p[2])))}"] = p[4].strip() if len(p) > 4 else ""
    return out


def fit_scale(sp: pd.DataFrame) -> dict:
    """pixel ~ array unit per axis (array units are 200 um apart) -> um/px and fit quality."""
    sx, ix = np.polyfit(sp["x"], sp["pixel_x"], 1)
    sy, iy = np.polyfit(sp["y"], sp["pixel_y"], 1)
    rx = np.sqrt(np.mean((sp["pixel_x"] - (sx * sp["x"] + ix)) ** 2))
    ry = np.sqrt(np.mean((sp["pixel_y"] - (sy * sp["y"] + iy)) ** 2))
    ppu = (abs(sx) + abs(sy)) / 2
    return dict(slope_x=sx, slope_y=sy, px_per_unit=ppu, um_per_px=K.LEGACY_PITCH_UM / ppu,
                xy_diff_pct=100 * abs(abs(sx) - abs(sy)) / ppu, rmse_px=(rx + ry) / 2,
                rmse_frac_pitch=(rx + ry) / 2 / ppu)


# ------------------------------------------------------------ gene symbols ----
class _IdentityDict(dict):
    """Stand-in for stnet.utils.ensembl.IdentityDict: unknown keys map to themselves."""
    def __missing__(self, key):
        return key


class _StnetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "IdentityDict":
            return _IdentityDict
        return super().find_class(module, name)


def ensembl_symbols(stnet_root: str | None = None) -> dict:
    """ENSG -> symbol with ST-Net's own table (so He symbols match what the ST-Net run used).
    ensembl.pkl pickles stnet's IdentityDict, so it is unpickled with a local stand-in class
    (no stnet / openslide import); without the pkl, stnet/utils/ensembl.py builds it from the tsv."""
    root = Path(stnet_root or os.environ.get("STNET", "/workspace/ST-Net"))
    pkl = root / "stnet" / "utils" / "ensembl.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            sym = _StnetUnpickler(f).load()
        return sym if isinstance(sym, _IdentityDict) else _IdentityDict(sym)
    spec = importlib.util.spec_from_file_location("stnet_ensembl", root / "stnet" / "utils" / "ensembl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.symbol


# ------------------------------------------------------------ her2st-like ----
def load_counts_npz(sec: str):
    """-> (csr spots x features, ENSG ids, symbols, spot ids)."""
    from scipy import sparse
    z = np.load(HE_CALIB / f"counts_{sec}.npz", allow_pickle=True)
    X = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return X, [str(g) for g in z["gene_ids"]], [str(g) for g in z["genes"]], [str(s) for s in z["spot_id"]]


def measured_mask(genes: list[str]) -> np.ndarray:
    """1 = panel gene appears in at least one He section's feature list (He, like her2st, drops
    genes with 0 counts per section, so per-section absence is a measured zero)."""
    p = HE_CALIB / "panel_coverage.tsv"
    cov = pd.read_csv(p, sep="\t")
    ok = set(cov.loc[cov["measured"] == 1, "gene"])
    return np.array([g in ok for g in genes])


def write_preds(out_dir: Path, sec: str, *, pred, truth, spot_ids, genes, trainmean,
                model: str, fold: int, extra: dict | None = None):
    """Same contract as common.write_preds (<out_dir>/preds/<SEC>.npz), cohort 'he' or 'her2st'.
    her2st held-out sections go through common.write_preds unchanged."""
    pred = np.asarray(pred, np.float32)
    truth = np.asarray(truth, np.float32)
    assert pred.shape == truth.shape == (len(spot_ids), len(genes)), (pred.shape, truth.shape)
    assert np.isfinite(pred).all(), f"{sec}: non-finite predictions"
    md = metadata().loc[sec]
    out = Path(out_dir) / "preds"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{sec}.npz", pred=pred, truth=truth, spot_id=np.array(spot_ids),
                        genes=np.array(genes), trainmean=np.asarray(trainmean, np.float32),
                        measured=measured_mask(list(genes)), section=sec, model=model, fold=int(fold),
                        cohort="he", patient=md["patient"], subtype=md["type"])
    if extra:
        (Path(out_dir) / "run.json").write_text(json.dumps(extra, indent=2, default=str))
