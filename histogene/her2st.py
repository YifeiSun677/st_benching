"""her2st I/O: section list, counts, spot files, gene panel, expression target, folds.

Reproduces the arithmetic of the original ViT_HER2ST dataset but
  (1) uses OUR 833-gene panel instead of the hard-coded 785 list,
  (2) zero-fills panel genes that a section's count matrix does not contain,
  (3) drops the scprep dependency (two lines of numpy, verified identical).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------- names -----
def section_names() -> list[str]:
    """All her2st section names, sorted, e.g. ['A1','A2',...,'H3'] (36 of them)."""
    files = sorted(os.path.basename(p) for p in glob.glob(str(C.CNT_DIR / "*.tsv*")))
    if not files:
        raise FileNotFoundError(f"no *.tsv / *.tsv.gz under {C.CNT_DIR}")
    return [f[:2] for f in files]


def _cnt_path(name: str) -> Path:
    for ext in (".tsv", ".tsv.gz"):
        p = C.CNT_DIR / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no count file for section {name} in {C.CNT_DIR}")


def _pos_path(name: str) -> Path:
    for ext in (".tsv", ".tsv.gz"):
        p = C.POS_DIR / f"{name}_selection{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no spot file for section {name} in {C.POS_DIR}")


def image_path(name: str) -> Path:
    d = C.IMG_DIR / name[0] / name
    files = sorted(os.listdir(d))
    if not files:
        raise FileNotFoundError(f"no image in {d}")
    return d / files[0]


# ------------------------------------------------------------------ meta -----
def read_meta(name: str) -> pd.DataFrame:
    """counts (spots x genes) joined with the spot file, indexed by '<x>x<y>' id.

    Same join as the original repo, plus a drop of spots that have no pixel
    coordinate (those would crash the int cast anyway).
    """
    cnt = pd.read_csv(_cnt_path(name), sep="\t", index_col=0)
    pos = pd.read_csv(_pos_path(name), sep="\t")
    x = np.around(pos["x"].values).astype(int)
    y = np.around(pos["y"].values).astype(int)
    pos["id"] = [f"{a}x{b}" for a, b in zip(x, y)]
    meta = cnt.join(pos.set_index("id"))
    before = len(meta)
    meta = meta.dropna(subset=["pixel_x", "pixel_y"])
    if len(meta) != before:
        print(f"  [{name}] dropped {before - len(meta)} spot(s) with no pixel coordinate")
    return meta


# ----------------------------------------------------------------- panel -----
def load_panel(path: Path | str | None = None) -> list[str]:
    path = Path(path or C.PANEL_FILE)
    genes = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(genes) != len(set(genes)):
        raise ValueError("panel file contains duplicate symbols")
    return genes


def expression(meta: pd.DataFrame, panel: list[str]) -> np.ndarray:
    """Target matrix: log10(CP10K + 1), computed over the panel columns only.

    This is exactly what the repo does via
        scprep.transform.log(scprep.normalize.library_size_normalize(X))
    since scprep defaults are rescale=10000, base=10, pseudocount=1.
    Genes absent from this section are zero-filled (they contribute 0 to the
    library size), so the matrix always has len(panel) columns in panel order.
    """
    present = [g for g in panel if g in meta.columns]
    X = np.zeros((len(meta), len(panel)), dtype=np.float64)
    idx = {g: i for i, g in enumerate(panel)}
    if present:
        sub = meta[present].values.astype(np.float64)
        for j, g in enumerate(present):
            X[:, idx[g]] = sub[:, j]
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    X = X / lib * 1e4
    return np.log10(X + 1.0).astype(np.float32)


def panel_coverage(panel: list[str]) -> pd.DataFrame:
    """How many of the 833 panel genes each section actually carries."""
    rows = []
    for name in section_names():
        cols = set(pd.read_csv(_cnt_path(name), sep="\t", index_col=0, nrows=1).columns)
        rows.append({"section": name,
                     "present": sum(g in cols for g in panel),
                     "missing": sum(g not in cols for g in panel)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- folds -----
def folds(cv: str | None = None) -> list[dict]:
    """Return [{fold, test:[sections], train:[sections]}, ...].

    cv='patient'  -> 8 folds, leave-one-PATIENT-out (A..H), all 36 sections used.
    cv='section'  -> the repo's own split: samples = names[1:33] (A2..G3), one
                     fold per section. NOTE this silently excludes A1 and all of
                     patient H from both train and test.
    """
    cv = cv or C.CV
    names = section_names()
    if cv == "patient":
        patients = sorted({n[0] for n in names})
        out = []
        for i, p in enumerate(patients):
            te = [n for n in names if n[0] == p]
            tr = [n for n in names if n[0] != p]
            out.append({"fold": i, "name": p, "test": te, "train": tr})
        return out
    if cv == "section":
        samples = names[1:33]
        return [{"fold": i, "name": s, "test": [s],
                 "train": [x for x in samples if x != s]}
                for i, s in enumerate(samples)]
    raise ValueError(f"unknown cv scheme {cv!r}")
