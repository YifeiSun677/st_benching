"""
her2st helpers, driven by the BLEEP cache's own metadata so TRIPLEX sees the
EXACT same patches, spot order, gene panel and ground-truth expression as every
other model in the benchmark.

The cache directory contains:
  patches.npy    (N,224,224,3) uint8   -- open with np.load(mmap_mode="r")
  expression.npy (N,833) float32       -- CPM + log1p target, row-aligned
  index.json     { sections: {sec:[start,end)}, spot_keys:["sec:AxB",...],
                   panel:[...833...], patch_size, n_spots, ... }

`spot_keys[i]` is the spot in cache row i, so it fixes the order end to end.
We derive the spot index (section, patient, spot_id, array_row/col, cache_idx)
from it -- no raw spotfiles, no join, no ordering guesswork. The target for a
spot is simply expression.npy at its cache row.
"""
import os
import json
import numpy as np
import pandas as pd

from . import config

_INDEX_CACHE = None
_EXPR_HANDLE = None


# ------------------------------------------------------------- index.json
def load_index_json(path=None):
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        path = path or config.BLEEP_INDEX_JSON
        with open(path) as f:
            _INDEX_CACHE = json.load(f)
    return _INDEX_CACHE


def canonical_panel():
    """The gene order of expression.npy -- the authoritative panel."""
    return list(load_index_json()["panel"])


def load_panel(path=None):
    """Return the canonical panel (from the cache), and warn if the repo's
    panel_833.txt disagrees, since expression.npy is column-aligned to the
    cache panel, not to the text file."""
    panel = canonical_panel()
    if len(panel) != config.NUM_GENES:
        print(f"[her2st] WARNING: cache panel has {len(panel)} genes, "
              f"config.NUM_GENES={config.NUM_GENES}")
    txt = path or config.PANEL_833
    if os.path.exists(txt):
        with open(txt) as f:
            file_panel = [ln.strip() for ln in f if ln.strip()]
        if file_panel != panel:
            same = set(file_panel) == set(panel)
            print(f"[her2st] NOTE: panel_833.txt "
                  f"{'reorders' if same else 'DIFFERS from'} the cache panel; "
                  f"using the cache panel (matches expression.npy).")
    return panel


# ------------------------------------------------------------- expression (truth)
def load_expression():
    global _EXPR_HANDLE
    if _EXPR_HANDLE is None:
        _EXPR_HANDLE = np.load(config.HER2ST_EXPRESSION, mmap_mode="r")
    return _EXPR_HANDLE


def smooth_expr(expr, array_rc):
    """3x3 array-neighbourhood mean of the (already CPM+log1p) target, for the
    reported smoothed variant only (config.SMOOTH). Headline uses raw expr."""
    rows = array_rc[:, 0].astype(int)
    cols = array_rc[:, 1].astype(int)
    out = np.empty_like(expr)
    for i in range(len(expr)):
        m = (np.abs(rows - rows[i]) <= 1) & (np.abs(cols - cols[i]) <= 1)
        out[i] = expr[m].mean(0)
    return out.astype(np.float32)


# ------------------------------------------------------------- spot index
def _parse_spot_key(key):
    """"A1:10x13" -> (section, spot_id, array_col, array_row)."""
    section, sid = key.split(":", 1)
    a, b = sid.lower().split("x")
    col = int(round(float(a)))   # her2st id is "Xarray x Yarray" = (col, row)
    row = int(round(float(b)))
    return section, sid, col, row


def build_spot_index(out_csv=None):
    """Derive the spot index CSV straight from index.json's spot_keys (cache
    order preserved: cache_idx == position in spot_keys)."""
    out_csv = out_csv or config.SPOT_INDEX
    j = load_index_json()
    keys = j["spot_keys"]
    ranges = j["sections"]
    rows = []
    for i, key in enumerate(keys):
        section, sid, col, row = _parse_spot_key(key)
        rows.append(dict(idx=i, section=section, patient=section[0],
                         spot_id=sid, array_row=row, array_col=col, cache_idx=i))
    df = pd.DataFrame(rows)
    # integrity: section blocks must match index.json's [start,end) ranges
    for section, (start, end) in ranges.items():
        blk = df[(df.cache_idx >= start) & (df.cache_idx < end)]
        assert (blk.section == section).all(), f"order mismatch in {section}"
        assert len(blk) == end - start, f"count mismatch in {section}"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[her2st] wrote spot index: {len(df)} spots, "
          f"{df.section.nunique()} sections -> {out_csv}")
    return df


def load_spot_index(path=None):
    path = path or config.SPOT_INDEX
    if not os.path.exists(path):
        # deterministic to rebuild from the cache metadata -- just do it
        if os.path.exists(config.BLEEP_INDEX_JSON):
            print(f"[her2st] spot index missing; building from "
                  f"{config.BLEEP_INDEX_JSON}")
            return build_spot_index(path)
        raise FileNotFoundError(
            f"Spot index not found at {path} and no index.json at "
            f"{config.BLEEP_INDEX_JSON} to build it from.")
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


if __name__ == "__main__":
    # convenience: build the spot index from index.json
    build_spot_index()
