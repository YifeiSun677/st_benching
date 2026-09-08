"""
her2st reading + target transform + 833-panel alignment + fold construction.

Faithful to ViT_HER2ST in the TCGN repo, with three documented deviations that
match the rest of the benchmark:
  * gene panel = the common 833 list (not upstream's 785 her_hvg_cut_1000);
  * genes absent from a section's count matrix are ZERO-FILLED so the panel
    stays length-833 across sections/folds (benchmark-wide decision);
  * target reimplemented in numpy (no scprep on the pod). Verified bit-exact
    against scprep 1.2.3: library_size_normalize(rescale=10000) -> log10(x+1).
"""
import os
import glob
import numpy as np
import pandas as pd

import config as C


# ---------------------------------------------------------------- file access
def _cnt_path(name):
    for ext in (".tsv", ".tsv.gz"):
        p = os.path.join(C.CNT_DIR, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("no count file for section %s in %s" % (name, C.CNT_DIR))


def list_sections():
    """All her2st sections present, e.g. ['A1','A2',...,'H3'], sorted."""
    files = sorted(os.listdir(C.CNT_DIR))
    names = [f[:2] for f in files if f.endswith(".tsv") or f.endswith(".tsv.gz")]
    # de-dup preserving order (A1.tsv and A1.tsv.gz would both map to 'A1')
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); out.append(n)
    return out


def get_cnt(name):
    df = pd.read_csv(_cnt_path(name), sep="\t", index_col=0)   # index = 'XxY' spot id
    return df


def get_pos(name):
    df = pd.read_csv(os.path.join(C.POS_DIR, name + "_selection.tsv"), sep="\t")
    x = np.around(df["x"].values).astype(int)
    y = np.around(df["y"].values).astype(int)
    df = df.copy()
    df["id"] = [str(x[i]) + "x" + str(y[i]) for i in range(len(x))]
    return df


def get_meta(name):
    """cnt joined with spotfile on the 'XxY' id (verbatim ViT_HER2ST join)."""
    cnt = get_cnt(name)
    pos = get_pos(name)
    meta = cnt.join(pos.set_index("id"))
    return meta


def get_img_path(name):
    pre = os.path.join(C.IMG_DIR, name[0], name)
    files = os.listdir(pre)
    return os.path.join(pre, files[0])


# ------------------------------------------------------------- target transform
def load_panel():
    with open(C.PANEL_FILE) as f:
        genes = [ln.strip() for ln in f if ln.strip()]
    assert len(genes) == 833, "expected 833 genes, got %d in %s" % (len(genes), C.PANEL_FILE)
    return genes


def counts_on_panel(meta, panel):
    """Reindex the section's gene columns to the 833 panel, missing -> 0."""
    # meta has gene columns (from cnt) plus x/y/pixel_x/... (from pos). Restrict
    # to genes by reindexing on the panel; absent genes come back as NaN -> 0.
    sub = meta.reindex(columns=panel)
    return sub.fillna(0.0).values.astype(np.float64)


def normalize_target(counts):
    """log_base( X / rowsum * rescale + psc ). Matches scprep defaults."""
    rowsum = counts.sum(axis=1, keepdims=True)
    rowsum = np.where(rowsum == 0, 1.0, rowsum)
    if C.TARGET_RESCALE == "median":
        rescale = np.median(counts.sum(axis=1))
    else:
        rescale = float(C.TARGET_RESCALE)
    norm = counts / rowsum * rescale
    base = {2: np.log2, 10: np.log10}.get(C.TARGET_LOG_BASE, np.log)
    return base(norm + C.TARGET_PSEUDOCOUNT)


def section_targets(name, panel):
    """Return (expr[N,833] float32, centers[N,2] int, spot_ids[N] str)."""
    meta = get_meta(name)
    counts = counts_on_panel(meta, panel)
    expr = normalize_target(counts).astype(np.float32)
    centers = np.floor(meta[["pixel_x", "pixel_y"]].values).astype(int)   # (px, py)
    spot_ids = np.asarray(meta.index.astype(str))
    return expr, centers, spot_ids


# ----------------------------------------------------------------------- folds
def patient_of(section):
    return section[0]


def lopo_folds():
    """Leave-one-PATIENT-out over all 8 patients A..H. Returns list of
    (test_patient, test_sections, train_sections)."""
    secs = list_sections()
    patients = sorted({patient_of(s) for s in secs})
    folds = []
    for p in patients:
        te = [s for s in secs if patient_of(s) == p]
        tr = [s for s in secs if patient_of(s) != p]
        folds.append((p, te, tr))
    return folds


def loso_folds():
    """Repo-native leave-one-SECTION-out over names[1:33] (A2..G3; drops A1 and
    all of H). Only for the reproduction sanity check, not the headline."""
    secs = list_sections()
    samples = secs[1:33]
    return [(s, [s], [t for t in samples if t != s]) for s in samples]
