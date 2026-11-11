"""
ridge/data.py -- spot index, raw counts, target transforms, labels, gene sets.

Everything is keyed by "SEC:XxY" (section + ST array coordinates), so arrays from
different ports can be aligned without trusting any file's row order.

Target transforms (verified against each port's stored truth on section A1,
max |diff| < 1e-6 unless noted; C = 833-panel counts, zero-filled):

  cpm_panel_ln         ln(1 + 1e6 * C / sum_panel(C))       shared expression.npy (BLEEP, TRIPLEX)
  cp10k_panel_ln       ln(1 + 1e4 * C / sum_panel(C))       STFlow normalised-target run
  cp10k_panel_log10    log10(1 + 1e4 * C / sum_panel(C))    HisToGene (row sums match to 5e-5)
  cp10k_full_log10     log10(1 + 1e4 * C / sum_all_genes)   DeepPT (row sums match to 2e-5)
  mediannorm_panel_ln  ln(1 + med_s * C / sum_panel(C))     Path2Space; med_s = median of the
                                                            section's non-zero panel depths
  raw_ln               ln(1 + C)                            STFlow upstream (old) run

log10 and ln differ by a constant factor, so PCC is identical between
cp10k_panel_ln and cp10k_panel_log10; only MAE/MSE differ.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from . import config


# ------------------------------------------------------------------ keys
def norm_spot_id(s) -> str:
    """'10x13', '10.0x13.0', 'A1:10x13', b'10x13' -> '10x13' (X = array col, Y = array row)."""
    if isinstance(s, bytes):
        s = s.decode()
    s = str(s).split(":")[-1].strip().lower()
    a, b = s.split("x")
    return f"{int(round(float(a)))}x{int(round(float(b)))}"


def make_key(section: str, spot_id) -> str:
    return f"{section}:{norm_spot_id(spot_id)}"


def load_panel(path: str | None = None) -> list[str]:
    with open(path or config.PANEL) as f:
        genes = [ln.strip() for ln in f if ln.strip()]
    assert len(genes) == len(set(genes)), "panel has duplicate genes"
    return genes


# ------------------------------------------------------------------ spot index
def _index_from_keys(keys, source: str) -> pd.DataFrame:
    rows = []
    for i, k in enumerate(keys):
        section, sid = str(k).split(":", 1)
        sid = norm_spot_id(sid)
        ax, ay = (int(v) for v in sid.split("x"))
        rows.append(dict(row=i, key=f"{section}:{sid}", section=section,
                         patient=section[0], spot_id=sid, ax=ax, ay=ay))
    df = pd.DataFrame(rows)
    assert df.key.is_unique, "duplicate spot keys in index"
    df.attrs["source"] = source
    return df


def _sections_from_counts(root: str) -> list[str]:
    files = glob.glob(os.path.join(root, "ST-cnts", "*.tsv*"))
    return sorted({os.path.basename(f).split(".tsv")[0] for f in files})


def read_counts_df(section: str, root: str | None = None) -> pd.DataFrame:
    root = root or config.HER2ST_ROOT
    for ext in (".tsv.gz", ".tsv"):
        p = os.path.join(root, "ST-cnts", f"{section}{ext}")
        if os.path.exists(p):
            df = pd.read_csv(p, sep="\t", index_col=0)
            df.index = [norm_spot_id(s) for s in df.index]
            return df
    raise FileNotFoundError(f"no counts file for {section} under {root}/ST-cnts")


def build_index(index_json: str | None = None, root: str | None = None) -> pd.DataFrame:
    """Canonical spot order. Prefer the shared cache's index.json (the order every
    cache-based port uses); otherwise fall back to the raw counts files."""
    p = index_json or config.INDEX_JSON
    if os.path.exists(p):
        with open(p) as f:
            j = json.load(f)
        return _index_from_keys(j["spot_keys"], source=f"index.json ({p})")
    root = root or config.HER2ST_ROOT
    keys = []
    for sec in _sections_from_counts(root):
        keys += [f"{sec}:{s}" for s in read_counts_df(sec, root).index]
    return _index_from_keys(keys, source=f"raw counts ({root})")


# ------------------------------------------------------------------ counts
def load_counts(index: pd.DataFrame, genes: list[str], root: str | None = None,
                cache_path: str | None = None, save: bool = True):
    """(counts_panel [n, G] float32, lib_full [n] float64), rows = index order.
    Genes missing from a section's file are zero-filled (benchmark-wide rule).
    lib_full = depth over ALL genes in the file (needed for DeepPT's target)."""
    cache_path = cache_path or os.path.join(config.RIDGE_CACHE, "counts.npz")
    if save and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if list(z["keys"]) == list(index.key) and list(z["genes"]) == list(genes):
            return z["counts"], z["lib_full"]
    counts = np.zeros((len(index), len(genes)), np.float32)
    lib_full = np.zeros(len(index), np.float64)
    for sec, grp in index.groupby("section", sort=False):
        df = read_counts_df(sec, root)
        missing = set(grp.spot_id) - set(df.index)
        assert not missing, f"{sec}: {len(missing)} index spots absent from counts"
        sub = df.loc[grp.spot_id.values]
        lib_full[grp.row.values] = sub.sum(1).values
        counts[grp.row.values] = sub.reindex(columns=genes, fill_value=0).values
    if save:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, counts=counts, lib_full=lib_full,
                            keys=index.key.to_numpy(dtype=str), genes=np.array(genes))
    return counts, lib_full


# ------------------------------------------------------------------ targets
def _depth_norm(C, depth, scale):
    depth = np.asarray(depth, np.float64).reshape(-1, 1)
    out = np.zeros_like(C, dtype=np.float64)
    np.divide(C, depth, out=out, where=depth > 0)          # zero-depth spot -> all zeros
    return out * (np.asarray(scale, np.float64).reshape(-1, 1) if np.ndim(scale) else scale)


def _median_scale(C, sections):
    lib = C.sum(1)
    scale = np.ones(len(C))
    for s in np.unique(sections):
        m = sections == s
        pos = lib[m][lib[m] > 0]
        scale[m] = np.median(pos) if pos.size else 1.0
    return scale


TARGETS = {
    "cpm_panel_ln":        lambda C, lf, sec: np.log1p(_depth_norm(C, C.sum(1), 1e6)),
    "cp10k_panel_ln":      lambda C, lf, sec: np.log1p(_depth_norm(C, C.sum(1), 1e4)),
    "cp10k_panel_log10":   lambda C, lf, sec: np.log10(1 + _depth_norm(C, C.sum(1), 1e4)),
    "cp10k_full_log10":    lambda C, lf, sec: np.log10(1 + _depth_norm(C, lf, 1e4)),
    "mediannorm_panel_ln": lambda C, lf, sec: np.log1p(_depth_norm(C, C.sum(1), _median_scale(C, sec))),
    "raw_ln":              lambda C, lf, sec: np.log1p(C.astype(np.float64)),
}


def make_target(name: str, counts, lib_full, sections) -> np.ndarray:
    """Target matrix in the named space. `sections` is needed for mediannorm
    (per-section median), so always pass whole sections."""
    if name not in TARGETS:
        raise KeyError(f"unknown target {name!r}; choose from {sorted(TARGETS)}")
    C = np.asarray(counts, np.float64)
    return TARGETS[name](C, np.asarray(lib_full), np.asarray(sections)).astype(np.float32)


def identify_target(truth, counts, lib_full, sections, row_order_known=True):
    """Which TARGETS entry reproduces `truth`? Returns [(name, max_abs_diff)] sorted.
    If row order is unknown (e.g. HisToGene), compares sorted row sums instead."""
    out = []
    for name in TARGETS:
        T = make_target(name, counts, lib_full, sections)
        if row_order_known:
            d = float(np.abs(T - truth).max())
        else:
            d = float(np.abs(np.sort(T.sum(1)) - np.sort(np.asarray(truth).sum(1))).max())
        out.append((name, d))
    return sorted(out, key=lambda t: t[1])


# ------------------------------------------------------------------ labels
def load_region_labels(index: pd.DataFrame, root: str | None = None) -> np.ndarray:
    """Pathologist label per spot (object array, None where unlabelled).
    Files exist for one section per patient: A1 B1 C1 D1 E1 F1 G2 H1 (J1 has no counts)."""
    root = root or config.HER2ST_ROOT
    labels = np.full(len(index), None, dtype=object)
    pos = dict(zip(index.key, index.row))
    for f in sorted(glob.glob(os.path.join(root, "ST-pat", "lbl", "*_labeled_coordinates.tsv"))):
        sec = os.path.basename(f).split("_")[0]
        df = pd.read_csv(f, sep="\t").dropna(subset=["x", "y", "label"])
        for x, y, lab in zip(df.x, df.y, df.label):
            r = pos.get(f"{sec}:{int(round(x))}x{int(round(y))}")
            if r is not None:
                labels[r] = str(lab)
    return labels


# ------------------------------------------------------------------ gene sets
def load_gene_sets(genes: list[str], path: str | None = None) -> dict[str, np.ndarray]:
    """{'all','hvg','svg','marker'} -> boolean masks over `genes` (missing files skipped)."""
    path = path or config.GENE_SETS
    out = {"all": np.ones(len(genes), bool)}
    for name in ("hvg", "svg", "marker"):
        p = os.path.join(path, f"gene_set_{name}.txt")
        if os.path.exists(p):
            with open(p) as f:
                s = {ln.strip() for ln in f if ln.strip()}
            out[name] = np.array([g in s for g in genes])
    return out


# ------------------------------------------------------------------ grid neighbours
def ring_neighbours(index: pd.DataFrame, r: int) -> list[np.ndarray]:
    """For each spot, the rows of same-section spots at Chebyshev distance exactly r
    on the ST array (r=1: the 8 spots of the 3x3 ring; r=2: the 16 of the 5x5 ring)."""
    out = [None] * len(index)
    offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
               if max(abs(dx), abs(dy)) == r]
    for _, grp in index.groupby("section", sort=False):
        at = {(x, y): row for x, y, row in zip(grp.ax, grp.ay, grp.row)}
        for x, y, row in zip(grp.ax, grp.ay, grp.row):
            out[row] = np.array([at[(x + dx, y + dy)] for dx, dy in offsets
                                 if (x + dx, y + dy) in at], dtype=np.int64)
    return out
