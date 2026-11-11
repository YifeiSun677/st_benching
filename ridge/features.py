"""
ridge/features.py -- read the frozen-feature caches the other ports already wrote,
align them to the canonical spot index, and build the context blocks.

Nothing here re-runs an encoder: each source is the exact representation a deep
model in the benchmark consumed, so a ridge on it is that model's "frozen twin".

    python -m ridge.features --sources uni_hest112 cigar resnet50 ctranspath
    python -m ridge.features --sources all          # every source whose cache exists
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from . import config, data


# ------------------------------------------------------------------ per-source readers
# Each reader yields (section, spot_ids, features[n, d]) for every file it finds.
def _read_stflow(tag):
    def reader():
        for f in sorted(glob.glob(os.path.join(config.STFLOW_CACHE, tag, "*.npz"))):
            z = np.load(f, allow_pickle=True)
            yield os.path.basename(f)[:-4], z["spot_id"], z["features"]
    return reader


def _read_triplex():
    import h5py
    for f in sorted(glob.glob(os.path.join(config.TRIPLEX_FEATURES, "*.h5"))):
        with h5py.File(f, "r") as h:
            yield os.path.basename(f)[:-3], h["spot_id"][()], h["features"][()]


def _read_deeppt():
    for f in sorted(glob.glob(os.path.join(config.DEEPPT_FEATURES, "*.npy"))):
        sec = os.path.basename(f)[:-4]
        spots = pd.read_csv(os.path.join(config.DEEPPT_FEATURES, f"{sec}_spots.csv"))
        yield sec, spots["spot_id"].values, np.load(f)


def _read_p2s():
    for f in sorted(glob.glob(os.path.join(config.P2S_FEATURES, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        yield os.path.basename(f)[:-4], z["spot_id"], z["feat"]


READERS = {
    "uni_hest112": _read_stflow("uni_v1_hest112"),
    "uni_native224": _read_stflow("uni_v1_native224"),
    "cigar": _read_triplex,
    "resnet50": _read_deeppt,
    "ctranspath": _read_p2s,
}


def aligned_path(source: str) -> str:
    return os.path.join(config.RIDGE_CACHE, f"features_{source}.npz")


def align_source(source: str, index: pd.DataFrame, reader=None):
    """Read one source and return (X [n_index, d] float32, report dict).
    Raises if any canonical spot is missing or a key is duplicated."""
    reader = reader or READERS[source]
    pos = dict(zip(index.key, index.row))
    X, seen, extra, n_files = None, np.zeros(len(index), bool), 0, 0
    for sec, sids, feats in reader():
        n_files += 1
        feats = np.asarray(feats, np.float32)
        assert len(sids) == len(feats), f"{source}/{sec}: {len(sids)} ids vs {len(feats)} rows"
        if X is None:
            X = np.zeros((len(index), feats.shape[1]), np.float32)
        for sid, v in zip(sids, feats):
            r = pos.get(data.make_key(sec, sid))
            if r is None:
                extra += 1
                continue
            assert not seen[r], f"{source}: duplicate spot {sec}:{sid}"
            X[r], seen[r] = v, True
    if X is None:
        raise FileNotFoundError(f"{source}: no feature files found")
    missing = index.key[~seen].tolist()
    report = dict(source=source, files=n_files, dim=int(X.shape[1]),
                  matched=int(seen.sum()), missing=len(missing), extra=extra,
                  missing_examples=missing[:5],
                  nonfinite=int((~np.isfinite(X[seen])).sum()))
    if missing:
        raise ValueError(f"{source}: {len(missing)} canonical spots have no features, "
                         f"e.g. {missing[:5]}")
    return X, report


def save_aligned(source: str, X, index: pd.DataFrame, report: dict):
    os.makedirs(config.RIDGE_CACHE, exist_ok=True)
    np.savez(aligned_path(source), X=X, keys=index.key.to_numpy(dtype=str),
             report=np.array(repr(report)))


def load_aligned(source: str, index: pd.DataFrame) -> np.ndarray:
    p = aligned_path(source)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing -- run: python -m ridge.features --sources {source}")
    z = np.load(p, allow_pickle=False)
    if list(z["keys"]) != list(index.key):
        raise ValueError(f"{p} was aligned to a different spot index; rebuild it")
    return z["X"]


# ------------------------------------------------------------------ context blocks
def section_means(X, index: pd.DataFrame) -> np.ndarray:
    """[n, d]: every spot gets the mean feature vector of its own section (image only)."""
    out = np.zeros_like(X)
    for _, grp in index.groupby("section", sort=False):
        out[grp.row.values] = X[grp.row.values].mean(0)
    return out


def ring_block(X, index: pd.DataFrame, r: int):
    """[n, d+1]: mean feature of the ring-r neighbours, plus the neighbour count.
    A spot with no neighbours in the ring gets its section mean (a neutral value);
    the count column lets ridge tell such spots apart."""
    neigh = data.ring_neighbours(index, r)
    sec_mean = section_means(X, index)
    out = np.empty((len(X), X.shape[1] + 1), np.float32)
    for i, nb in enumerate(neigh):
        out[i, :-1] = X[nb].mean(0) if len(nb) else sec_mean[i]
        out[i, -1] = len(nb)
    return out


def static_blocks(X, index: pd.DataFrame, blocks: list[str]) -> dict[str, np.ndarray]:
    """Blocks that do not depend on the fold. 'global' is returned as raw section
    means; it is PCA-reduced inside each fold, on training sections only."""
    out = {}
    if "local" in blocks:
        out["local"] = X
    if "ring1" in blocks:
        out["ring1"] = ring_block(X, index, 1)
    if "ring2" in blocks:
        out["ring2"] = ring_block(X, index, 2)
    if "global" in blocks:
        out["global"] = section_means(X, index)
    return out


def global_pca(sec_means: np.ndarray, index: pd.DataFrame, train_mask: np.ndarray, k: int):
    """Fit PCA on one vector per TRAINING section, project every spot's section mean.
    Returns [n, k]. Unsupervised and image-only; the test patient is not used."""
    tr = index[train_mask].drop_duplicates("section")
    G = sec_means[tr.row.values].astype(np.float64)
    mu = G.mean(0)
    _, _, vt = np.linalg.svd(G - mu, full_matrices=False)
    k = min(k, vt.shape[0])
    return ((sec_means - mu) @ vt[:k].T).astype(np.float32)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["all"])
    args = ap.parse_args()
    index = data.build_index()
    print(f"[features] canonical index: {len(index)} spots from {index.attrs['source']}")
    sources = list(READERS) if args.sources == ["all"] else args.sources
    for s in sources:
        try:
            X, rep = align_source(s, index)
        except (FileNotFoundError, ValueError, AssertionError) as e:
            print(f"[features] {s}: SKIPPED -- {e}")
            continue
        save_aligned(s, X, index, rep)
        print(f"[features] {s}: {rep} -> {aligned_path(s)}")


if __name__ == "__main__":
    main()
