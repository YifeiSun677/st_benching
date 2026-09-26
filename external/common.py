"""external/common.py -- shared helpers for the her2st -> HER2+ Visium cross-cohort arm.

Everything model-agnostic lives here: paths, 10x readers, her2st readers, the
panel, pseudo-spot aggregation, and the ONE output contract every model driver
writes (write_preds).  Model drivers import this; they never re-implement it.

Data layout (decided in the runbook, Stage 2):
  /workspace/her2st/data            real her2st, untouched
  /workspace/ext/raw/<SEC>/         10x downloads (h5, spatial/, full-res tif)
  /workspace/ext/visium_like/data   Visium sections I1 I2 J1 K1 in her2st layout
                                    (kept SEPARATE from her2st: several ports list
                                    sections by globbing ST-cnts and build LOPO
                                    folds from that list, so mixing the trees would
                                    put Visium sections into training folds)
  /workspace/ext/calib/             scale tables, coverage, counts npz, pseudo-spots
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(os.environ.get("WS", "/workspace"))
EXT = Path(os.environ.get("EXT_ROOT", WS / "ext"))
RAW = EXT / "raw"
VIS_ROOT = EXT / "visium_like" / "data"
CALIB = EXT / "calib"
HER2ST_ROOT = Path(os.environ.get("HER2ST_ROOT", WS / "her2st" / "data"))
ST_BENCH = Path(os.environ.get("ST_BENCH", WS / "st_benching"))
PANEL_FILE = Path(os.environ.get("PANEL", ST_BENCH / "panels" / "panel_833.txt"))
GENE_SETS = Path(os.environ.get("GENE_SETS", ST_BENCH / "results" / "gene_sets"))

VISIUM_SECTIONS = ["I1", "I2", "J1", "K1"]
PATIENT_OF = {"I1": "I", "I2": "I", "J1": "J", "K1": "K"}
TIER_OF = {"I": 1, "J": 2, "K": 2}
HER2ST_PATIENTS = list("ABCDEFGH")
LEGACY_PITCH_UM = 200.0      # her2st array units
VISIUM_SPOT_UM = 55.0
VISIUM_PITCH_UM = 100.0      # centre-to-centre


# ------------------------------------------------------------------ panel ----
def load_panel(path: Path | str | None = None) -> list[str]:
    genes = [ln.strip().split("\t")[0] for ln in Path(path or PANEL_FILE).read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert len(genes) == len(set(genes)), "duplicate genes in panel"
    return genes


def load_gene_set(name: str) -> list[str]:
    p = GENE_SETS / f"gene_set_{name}.txt"
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


# ------------------------------------------------------------- 10x readers ----
def find_raw_files(sec: str) -> dict:
    d = RAW / sec
    h5 = sorted(glob.glob(str(d / "*filtered_feature_bc_matrix.h5")))
    sf = d / "spatial" / "scalefactors_json.json"
    pos = [p for p in (d / "spatial" / "tissue_positions.csv",
                       d / "spatial" / "tissue_positions_list.csv") if p.exists()]
    tifs = [p for p in glob.glob(str(d / "*.tif")) + glob.glob(str(d / "*.tiff"))
            if "/spatial/" not in p]
    tifs.sort(key=os.path.getsize, reverse=True)          # full-res image = the largest tif
    missing = [k for k, v in {"h5": h5, "scalefactors": sf.exists(), "positions": pos,
                              "image": tifs}.items() if not v]
    if missing:
        raise FileNotFoundError(f"{sec}: missing {missing} under {d}")
    return {"h5": Path(h5[0]), "scalefactors": sf, "positions": pos[0], "image": Path(tifs[0])}


def read_positions(path: Path) -> pd.DataFrame:
    """Space Ranger v1 (no header) or v2 (header) tissue positions -> one schema."""
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    first = Path(path).read_text().splitlines()[0]
    if first.startswith("barcode"):
        df = pd.read_csv(path)
        df.columns = cols
    else:
        df = pd.read_csv(path, header=None, names=cols)
    return df


def read_scalefactors(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def read_10x_h5(path: Path):
    """Returns (csr barcodes x genes, barcodes, gene_names, gene_ids), Gene Expression only."""
    import h5py
    from scipy import sparse
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][:])                       # (n_features, n_barcodes)
        X = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        barcodes = [b.decode() for b in m["barcodes"][:]]
        feat = m["features"]
        names = [b.decode() for b in feat["name"][:]]
        ids = [b.decode() for b in feat["id"][:]]
        ftype = [b.decode() for b in feat["feature_type"][:]]
    keep = np.array([t == "Gene Expression" for t in ftype])
    X = X[keep].T.tocsr()
    names = [n for n, k in zip(names, keep) if k]
    ids = [i for i, k in zip(ids, keep) if k]
    return X, barcodes, names, ids


def neighbour_distance_px(pos: pd.DataFrame, step=(0, 2)) -> float:
    """Median full-res pixel distance between (row, col) and (row+dr, col+dc).
    Every hex neighbour is VISIUM_PITCH_UM = 100 um away: step (0, 2) is the same-row
    neighbour, (1, 1) the diagonal one."""
    dr, dc = step
    key = {(r, c): (y, x) for r, c, y, x in pos[["array_row", "array_col", "pxl_row", "pxl_col"]].values}
    d = [np.hypot(y2 - y1, x2 - x1) for (r, c), (y1, x1) in key.items()
         if (r + dr, c + dc) in key for (y2, x2) in [key[(r + dr, c + dc)]]]
    return float(np.median(d))


def visium_scale(pos: pd.DataFrame, sf: dict) -> dict:
    """um/px from the 100-um spot pitch (primary).  Two sanity checks:
    * row vs diagonal neighbour distance agree within 2 % (regular hex grid, no skew)
    * implied spot_diameter_fullres in um lies in 60-70 um -- 10x defines that field as a
      VISUALISATION diameter of 60-70 um depending on slide design, NOT the physical 55 um."""
    row_px = neighbour_distance_px(pos, (0, 2))
    diag_px = neighbour_distance_px(pos, (1, 1))
    umpp = VISIUM_PITCH_UM / row_px
    return dict(um_per_px=umpp, row_px=row_px, diag_px=diag_px,
                hex_skew_pct=100 * abs(row_px - diag_px) / row_px,
                implied_spot_diam_um=sf["spot_diameter_fullres"] * umpp)


# ------------------------------------------------------------ her2st-like ----
def read_counts(sec: str, root: Path | None = None) -> pd.DataFrame:
    root = Path(root or (VIS_ROOT if sec in VISIUM_SECTIONS else HER2ST_ROOT))
    for ext in (".tsv.gz", ".tsv"):
        p = root / "ST-cnts" / f"{sec}{ext}"
        if p.exists():
            return pd.read_csv(p, sep="\t", index_col=0)
    raise FileNotFoundError(f"no counts for {sec} under {root}")


def read_spots(sec: str, root: Path | None = None) -> pd.DataFrame:
    root = Path(root or (VIS_ROOT if sec in VISIUM_SECTIONS else HER2ST_ROOT))
    for ext in (".tsv", ".tsv.gz"):
        p = root / "ST-spotfiles" / f"{sec}_selection{ext}"
        if p.exists():
            df = pd.read_csv(p, sep="\t")
            df.index = [f"{int(round(a))}x{int(round(b))}" for a, b in zip(df["x"], df["y"])]
            return df
    raise FileNotFoundError(f"no spot file for {sec} under {root}")


def her2st_sections(root: Path | None = None) -> list[str]:
    files = glob.glob(str(Path(root or HER2ST_ROOT) / "ST-cnts" / "*.tsv*"))
    return sorted(os.path.basename(f)[:2] for f in files)


def lopo_folds() -> list[dict]:
    """Same 8 folds as every port: fold k holds out patient HER2ST_PATIENTS[k]."""
    secs = her2st_sections()
    return [{"fold": k, "patient": p, "test": [s for s in secs if s[0] == p],
             "train": [s for s in secs if s[0] != p]} for k, p in enumerate(HER2ST_PATIENTS)]


def measured_mask(sec: str, genes: list[str]) -> np.ndarray:
    """1 = panel gene is in the sample's feature list. her2st sections: all True
    (the benchmark zero-fills sparse genes there; SD==0 genes are dropped at scoring)."""
    if sec not in VISIUM_SECTIONS:
        return np.ones(len(genes), bool)
    cov = pd.read_csv(CALIB / "panel_coverage.tsv", sep="\t")
    cov = cov[cov["section"] == sec].set_index("gene")["measured"]
    return np.array([bool(cov.get(g, 0)) for g in genes])


# ------------------------------------------------------------ pseudo-spots ----
def load_pseudo(sec: str) -> dict:
    z = np.load(CALIB / f"pseudospots_{sec}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def aggregate_counts(raw: np.ndarray, spot_ids: list[str], sec: str) -> tuple[np.ndarray, list[str]]:
    """Sum raw counts (spots x genes, rows in `spot_ids` order) over each 7-spot group.
    Returns (groups x genes summed counts, centre spot ids)."""
    ps = load_pseudo(sec)
    row = {s: i for i, s in enumerate(spot_ids)}
    members = ps["member_ids"]                               # (n_groups, 7) spot ids, centre first
    missing = {m for grp in members for m in grp if m not in row}
    if missing:
        raise KeyError(f"{sec}: {len(missing)} pseudo-spot members not in the prediction rows")
    idx = np.vectorize(row.__getitem__)(members)
    return raw[idx].sum(axis=1), list(members[:, 0])


# ---------------------------------------------------------------- contract ----
def write_preds(out_dir: Path, sec: str, *, pred, truth, spot_ids, genes, trainmean,
                truth_ps=None, centre_ids=None, model: str, fold: int, extra: dict | None = None):
    """The one output file every driver writes: <out_dir>/preds/<sec>.npz.

    pred, truth   (spots x 833) in the MODEL'S OWN target space, rows = spot_ids
    trainmean     (833,) mean of that fold's 7 training patients' truth, same space
    truth_ps      (groups x 833) summed-then-transformed truth; Visium only
    centre_ids    spot id of each group's centre (so pred_ps = pred[centre rows])
    """
    pred = np.asarray(pred, np.float32)
    truth = np.asarray(truth, np.float32)
    assert pred.shape == truth.shape == (len(spot_ids), len(genes)), (pred.shape, truth.shape)
    assert np.isfinite(pred).all(), f"{sec}: non-finite predictions"
    out = Path(out_dir) / "preds"
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(pred=pred, truth=truth, spot_id=np.array(spot_ids), genes=np.array(genes),
                   trainmean=np.asarray(trainmean, np.float32),
                   measured=measured_mask(sec, list(genes)),
                   section=sec, model=model, fold=int(fold),
                   cohort="visium" if sec in VISIUM_SECTIONS else "her2st")
    if truth_ps is not None:
        pos = {s: i for i, s in enumerate(spot_ids)}
        payload["truth_ps"] = np.asarray(truth_ps, np.float32)
        payload["centre_idx"] = np.array([pos[c] for c in centre_ids])
    np.savez_compressed(out / f"{sec}.npz", **payload)
    if extra:
        (Path(out_dir) / "run.json").write_text(json.dumps(extra, indent=2, default=str))
