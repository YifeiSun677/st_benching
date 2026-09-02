import numpy as np, pandas as pd
from . import config as C


def list_sections():
    names = []
    for p in sorted(C.CNT_DIR.iterdir()):
        n = p.name
        if n.endswith(".tsv") or n.endswith(".tsv.gz"):
            names.append(n.split(".")[0])
    return sorted(set(names))


def load_panel():
    genes = [g.strip() for g in open(C.PANEL_FILE) if g.strip()]
    assert len(genes) == len(set(genes)), "panel contains duplicate genes"
    return genes


def _cnt_path(name):
    for suf in (".tsv", ".tsv.gz"):
        p = C.CNT_DIR / f"{name}{suf}"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def load_section(name, panel):
    """One section -> dict. Row order = intersection of spotfile and cnts, sorted by spot id."""
    cnt = pd.read_csv(_cnt_path(name), sep="\t", index_col=0)
    pos = pd.read_csv(C.POS_DIR / f"{name}_selection.tsv", sep="\t")
    pos["id"] = pos["x"].astype(str) + "x" + pos["y"].astype(str)
    pos = pos.set_index("id")

    ids = sorted(set(cnt.index) & set(pos.index))
    cnt, pos = cnt.loc[ids], pos.loc[ids]

    # 833 panel, missing genes zero-filled (benchmark-wide decision)
    ori = cnt.reindex(columns=panel).fillna(0.0).values.astype(np.float64)

    lib = ori.sum(1)
    lib[lib == 0] = 1.0
    if C.NORM == "median":
        scale = np.median(lib)
    elif C.NORM == "cp10k":
        scale = 1e4
    else:
        raise ValueError(C.NORM)
    exp = np.log10(ori / lib[:, None] * scale + 1.0)

    sf = lib / np.median(lib)                       # ZINB size factor (same source as ori)

    return dict(
        name=name,
        spot_id=np.array(ids),
        array=pos[["x", "y"]].values.astype(np.int64),          # grid coords: pos-embedding + Grid pruning
        pixel=np.floor(pos[["pixel_x", "pixel_y"]].values).astype(np.int64),
        exp=exp.astype(np.float32),
        ori=ori.astype(np.float32),
        sf=sf.astype(np.float32),
        n_missing=int(sum(g not in cnt.columns for g in panel)),
    )
