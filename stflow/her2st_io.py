"""
stflow_port/her2st_io.py -- read raw her2st (Andersson et al. 2021 layout) directly.

Expected layout under HER2ST_ROOT:
    ST-cnts/<SEC>.tsv[.gz]              spots x genes raw counts, index = 'XxY' (e.g. '10x13')
    ST-spotfiles/<SEC>_selection.tsv    x, y, new_x, new_y, pixel_x, pixel_y[, selected]
    ST-imgs/<P>/<SEC>/*.jpg             one H&E image per section

Conventions (all checked in preflight.py):
  * spot id = f"{x}x{y}" built from the INTEGER x/y columns (not new_x/new_y), matching
    the ST-cnts index;
  * rows are kept in ST-cnts order, restricted to spots that also have a spot-file row;
  * pixel_x is the image COLUMN, pixel_y the image ROW (crop = img[y-r:y+r, x-r:x+r]);
  * target = log1p(raw count) on the fixed panel, genes missing from a section are
    zero-filled (benchmark-wide rule), which is STFlow's own `normalize_method=log1p`
    (sc.pp.log1p on raw counts, no library-size normalisation).
"""
import glob
import os
import re

import numpy as np
import pandas as pd

import config as C

SEC_RE = re.compile(r"^([A-H])(\d)$")


def list_sections(root=None):
    root = root or C.HER2ST_ROOT
    d = os.path.join(root, "ST-cnts")
    names = set()
    for f in os.listdir(d):
        stem = f.split(".")[0]
        if SEC_RE.match(stem) and ".tsv" in f:
            names.add(stem)
    return sorted(names)


def patient_of(section):
    return section[0]


def counts_path(section, root=None):
    root = root or C.HER2ST_ROOT
    hits = sorted(glob.glob(os.path.join(root, "ST-cnts", f"{section}.tsv*")))
    if not hits:
        raise FileNotFoundError(f"no ST-cnts file for {section}")
    return hits[0]


def spotfile_path(section, root=None):
    root = root or C.HER2ST_ROOT
    hits = sorted(glob.glob(os.path.join(root, "ST-spotfiles", f"{section}_selection.tsv*")))
    if not hits:
        raise FileNotFoundError(f"no ST-spotfiles/{section}_selection.tsv")
    return hits[0]


def image_path(section, root=None):
    root = root or C.HER2ST_ROOT
    pats = ["*.jpg", "*.JPG", "*.jpeg", "*.tif", "*.tiff", "*.png"]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(root, "ST-imgs", section[0], section, p)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no image under ST-imgs/{section[0]}/{section}/")


def read_counts(section, root=None):
    df = pd.read_csv(counts_path(section, root), sep="\t", index_col=0)
    df.index = df.index.astype(str)
    if df.columns.duplicated().any():  # sum duplicated gene columns, if any
        df = df.T.groupby(level=0).sum().T
    return df


def read_spots(section, root=None):
    raw = pd.read_csv(spotfile_path(section, root), sep="\t")
    cols = {c.lower(): c for c in raw.columns}
    for need in ("x", "y", "pixel_x", "pixel_y"):
        if need not in cols:
            raise ValueError(f"{section}: spot file lacks column '{need}' (has {list(raw.columns)})")
    x = raw[cols["x"]].round().astype(int)
    y = raw[cols["y"]].round().astype(int)
    out = pd.DataFrame({
        "x": x.values, "y": y.values,
        "new_x": raw[cols.get("new_x", cols["x"])].astype(float).values,
        "new_y": raw[cols.get("new_y", cols["y"])].astype(float).values,
        "pixel_x": raw[cols["pixel_x"]].astype(float).values,
        "pixel_y": raw[cols["pixel_y"]].astype(float).values,
    }, index=[f"{a}x{b}" for a, b in zip(x, y)])
    if "selected" in cols:
        out["selected"] = raw[cols["selected"]].values
    out = out[~out.index.duplicated(keep="first")]
    return out


def load_section(section, root=None):
    """Returns (counts_df aligned, spots_df aligned); both indexed by the same spot ids,
    in ST-cnts row order."""
    cnt = read_counts(section, root)
    spt = read_spots(section, root)
    keep = [s for s in cnt.index if s in spt.index]
    return cnt.loc[keep], spt.loc[keep]


def estimate_um_per_px(spots):
    """Least-squares slope of pixel coordinate on array coordinate, per axis.
    Array units are SPOT_PITCH_UM apart, so um/px = pitch / (px per array unit)."""
    sx = np.polyfit(spots["new_x"].values, spots["pixel_x"].values, 1)[0]
    sy = np.polyfit(spots["new_y"].values, spots["pixel_y"].values, 1)[0]
    px_per_unit = (abs(sx) + abs(sy)) / 2
    return C.SPOT_PITCH_UM / px_per_unit, abs(sx), abs(sy)


def load_panel(path=None):
    path = path or C.PANEL
    genes = []
    with open(path) as f:
        for line in f:
            g = line.strip().split("\t")[0].split(",")[0].strip()
            if g and not g.startswith("#"):
                genes.append(g)
    if genes and genes[0].lower() in ("gene", "genes", "symbol", "gene_name", "x"):
        genes = genes[1:]
    return genes


def compute_targets(counts_df, genes):
    """log1p(raw counts) on `genes`, zero-filled for genes absent from the section.
    Returns (labels float32 [N, G], raw float32 [N, G], n_missing)."""
    raw = counts_df.reindex(columns=genes, fill_value=0).to_numpy(dtype=np.float32)
    n_missing = int(sum(g not in counts_df.columns for g in genes))
    return np.log1p(raw).astype(np.float32), raw, n_missing
