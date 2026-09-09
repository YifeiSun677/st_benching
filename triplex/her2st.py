"""
her2st helpers: panel, spot index, per-section counts, target transform.

Contract with the shared cache
------------------------------
`SPOT_INDEX` (config.SPOT_INDEX) is a CSV with one row per spot and columns:
    idx, section, patient, spot_id, array_row, array_col, cache_idx
`cache_idx` indexes rows of the uint8 memmap at config.HER2ST_CACHE. Target,
global and neighbour features are all taken from those same rows, so TRIPLEX
sees exactly the patches every other model in the benchmark saw.

If you built the BLEEP 224px cache, you almost certainly already have an
equivalent index -- point SPOT_INDEX at it (rename columns if needed). If not,
build_spot_index_from_raw() will produce one from raw her2st spotfiles, but its
row order must then match how you built the cache.
"""
import os
import glob
import numpy as np
import pandas as pd

from . import config


def load_panel(path=None):
    path = path or config.PANEL_833
    with open(path) as f:
        genes = [ln.strip() for ln in f if ln.strip()]
    if len(genes) != config.NUM_GENES:
        print(f"[her2st] WARNING: panel has {len(genes)} genes, "
              f"config.NUM_GENES={config.NUM_GENES}")
    return genes


def load_spot_index(path=None):
    path = path or config.SPOT_INDEX
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Spot index not found at {path}. Either point SPOT_INDEX at your "
            f"BLEEP-cache index, or run build_spot_index_from_raw().")
    df = pd.read_csv(path, dtype={"spot_id": str})
    req = {"section", "patient", "spot_id", "array_row", "array_col", "cache_idx"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Spot index missing columns: {missing}")
    return df


def sections_for_patient(index_df, patient):
    return sorted(index_df.loc[index_df.patient == patient, "section"].unique())


def all_sections(index_df):
    return sorted(index_df.section.unique())


# ---------------------------------------------------------------- counts
def _counts_path(section):
    for ext in (".tsv.gz", ".tsv", ".csv.gz", ".csv"):
        p = os.path.join(config.HER2ST_ROOT, "ST-cnts", f"{section}{ext}")
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(config.HER2ST_ROOT, "**", f"{section}.tsv*"),
                     recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No counts file for section {section} under "
                            f"{config.HER2ST_ROOT}/ST-cnts")


def load_counts(section, panel, spot_ids):
    """
    Return raw counts (n_spots x 833) aligned to `panel`, in the order of
    `spot_ids`, zero-filling genes absent from this section (benchmark-wide
    decision so the panel length stays fixed at 833).
    """
    sep = "\t" if ".tsv" in _counts_path(section) else ","
    df = pd.read_csv(_counts_path(section), sep=sep, index_col=0)
    df.index = df.index.astype(str)
    # keep only panel genes that exist here; reindex to full panel (0-fill)
    df = df.reindex(columns=panel, fill_value=0.0)
    df = df.reindex(index=list(spot_ids), fill_value=0.0)
    return df.values.astype(np.float32)


def normalize_expr(counts, array_rc=None, cpm=config.CPM, smooth=config.SMOOTH):
    """
    Reproduces upstream normalize_adata: CPM(1e4) + log1p, optional 3x3
    array-neighbourhood mean smoothing. Smoothing is off by default (headline);
    turn it on for the reported smoothed variant.
    """
    x = counts.astype(np.float64)
    if cpm:
        tot = x.sum(1, keepdims=True)
        tot[tot == 0] = 1.0
        x = x / tot * 1e4
    x = np.log1p(x)
    if smooth:
        if array_rc is None:
            raise ValueError("smooth=True needs array_row/array_col")
        rows = array_rc[:, 0].astype(int)
        cols = array_rc[:, 1].astype(int)
        out = np.empty_like(x)
        for i in range(len(x)):
            m = (np.abs(rows - rows[i]) <= 1) & (np.abs(cols - cols[i]) <= 1)
            out[i] = x[m].mean(0)
        x = out
    return x.astype(np.float32)


# ---------------------------------------------------------------- optional raw index builder
def build_spot_index_from_raw(out_csv=None):
    """
    Best-effort spot index from raw her2st spotfiles (ST-spotfiles/*_selection.tsv).
    ONLY correct if your cache rows are laid out section-by-section (sections
    sorted) and spot-by-spot in spotfile order. If you built the cache
    differently, adapt this or reuse your existing index instead.
    """
    out_csv = out_csv or config.SPOT_INDEX
    files = sorted(glob.glob(os.path.join(config.HER2ST_ROOT,
                                          "ST-spotfiles", "*_selection.tsv")))
    if not files:
        raise FileNotFoundError("No ST-spotfiles/*_selection.tsv found.")
    rows, cache_idx = [], 0
    for fp in files:
        section = os.path.basename(fp).replace("_selection.tsv", "")
        patient = section[0]
        sf = pd.read_csv(fp, sep="\t")
        # her2st selection files carry x,y (array) and pixel_x,pixel_y; the
        # 'XxY' spot id uses array x/y.
        xcol = "x" if "x" in sf.columns else "new_x"
        ycol = "y" if "y" in sf.columns else "new_y"
        for _, r in sf.iterrows():
            ax, ay = int(round(r[xcol])), int(round(r[ycol]))
            rows.append(dict(idx=cache_idx, section=section, patient=patient,
                             spot_id=f"{ax}x{ay}", array_row=ay, array_col=ax,
                             cache_idx=cache_idx))
            cache_idx += 1
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[her2st] wrote {len(df)} spots -> {out_csv}")
    return df
