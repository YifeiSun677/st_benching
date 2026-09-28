"""
Target construction for the STFlow normalised-target run.

Two modes, so the old raw-target run stays reproducible from the same code path:

  raw_log1p           log1p(c_ig)                                  <- STFlow upstream / old run
  panel_cp10k_log1p   log1p(c_ig / d_i^panel * 1e4)                <- new run

where d_i^panel = sum over the 833 panel genes of c_ig, computed AFTER panel
subsetting and zero-filling, per spot, identically for train and held-out spots.

This must stay byte-for-byte equivalent to the definition used by
`score.py --renorm panel_cp10k`. `preflight_norm.py` checks that numerically
against the old run's stored truth; do not edit one without the other.

Natural log throughout (np.log1p), never log10.
"""
from __future__ import annotations

import numpy as np

CP10K = 1.0e4
MODES = ("raw_log1p", "panel_cp10k_log1p")

# Spots whose panel depth is 0 would divide by zero. Clamping the denominator to
# MIN_DEPTH leaves such a row as all zeros, which is what a 0-count spot should
# map to. Change this only if score.py does something different.
MIN_DEPTH = 1.0


def _as_counts(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts)
    if c.ndim != 2:
        raise ValueError(f"counts must be 2-D (spots, genes); got shape {c.shape}")
    c = c.astype(np.float64, copy=False)
    if not np.isfinite(c).all():
        raise ValueError("counts contain NaN/Inf")
    if (c < 0).any():
        raise ValueError("counts contain negative values")
    return c


def panel_depth(counts: np.ndarray) -> np.ndarray:
    """Per-spot total over the panel genes only. Shape (n_spots,)."""
    return _as_counts(counts).sum(axis=1)


def raw_log1p(counts: np.ndarray) -> np.ndarray:
    return np.log1p(_as_counts(counts)).astype(np.float32)


def panel_cp10k_log1p(counts: np.ndarray, min_depth: float = MIN_DEPTH) -> np.ndarray:
    c = _as_counts(counts)
    d = np.maximum(c.sum(axis=1, keepdims=True), min_depth)
    return np.log1p(c / d * CP10K).astype(np.float32)


def build_target(counts: np.ndarray, mode: str = "panel_cp10k_log1p",
                 min_depth: float = MIN_DEPTH) -> np.ndarray:
    """Single entry point used by the training script."""
    if mode == "raw_log1p":
        return raw_log1p(counts)
    if mode == "panel_cp10k_log1p":
        return panel_cp10k_log1p(counts, min_depth=min_depth)
    raise ValueError(f"unknown target mode {mode!r}; expected one of {MODES}")


def invert_raw_log1p(y: np.ndarray, round_to_int: bool = True) -> np.ndarray:
    """Recover raw counts from a raw_log1p target (exact up to float error,
    because the raw target is log1p of integers). Used by preflight to rebuild
    the new target from the old run's stored truth without touching her2st."""
    c = np.expm1(np.asarray(y, dtype=np.float64))
    if round_to_int:
        resid = np.abs(c - np.rint(c)).max()
        if resid > 1e-2:
            raise ValueError(
                f"expm1(truth) is not integral (max residual {resid:.3g}); "
                "the array passed in is probably not a raw log1p target")
        c = np.rint(c)
    return c


def target_report(counts: np.ndarray, mode: str) -> dict:
    """Cheap descriptive stats to print in preflight and write into run.json."""
    c = _as_counts(counts)
    y = build_target(c, mode=mode)
    d = c.sum(axis=1)
    return {
        "mode": mode,
        "n_spots": int(y.shape[0]),
        "n_genes": int(y.shape[1]),
        "n_zero_depth_spots": int((d <= 0).sum()),
        "panel_depth_median": float(np.median(d)),
        "panel_depth_min": float(d.min()),
        "panel_depth_max": float(d.max()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "target_sd": float(y.std()),
        "frac_zero": float((y == 0).mean()),
    }


def assert_sane(y: np.ndarray, mode: str) -> None:
    """Loud guard to call once inside train(), so a mis-wired target fails at
    step 0 rather than 90 epochs later."""
    y = np.asarray(y)
    if y.min() < 0:
        raise AssertionError(f"target has negative values (min {y.min():.4g})")
    if mode == "panel_cp10k_log1p":
        # log1p(1e4) = 9.21 is the hard ceiling (a spot with one expressed gene).
        if y.max() > 9.22:
            raise AssertionError(
                f"panel_cp10k target max {y.max():.4g} exceeds log1p(1e4)=9.2104 "
                "- the normalisation denominator is wrong")
        if y.max() < 3.0:
            raise AssertionError(
                f"panel_cp10k target max {y.max():.4g} is implausibly low "
                "- looks like a raw log1p target slipped through")
    if mode == "raw_log1p" and y.max() > 9.22:
        raise AssertionError(f"raw log1p target max {y.max():.4g} is implausible")
