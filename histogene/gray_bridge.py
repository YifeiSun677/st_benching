"""Grayscale for HisToGene, using the SAME code object as the BLEEP ablation.

Why a bridge instead of a copy
------------------------------
The requirement is that HisToGene's grayscale is the identical formula used by
the existing perturbation control (``bleep/gray.py``: ITU-R BT.601 luma,
0.299 R + 0.587 G + 0.114 B, float64 elementwise, rint, uint8 in / uint8 out).
A copied function can drift. So this module does not re-implement anything: it
loads ``<repo>/bleep/gray.py`` by file path and calls its ``apply_gray_uint8``.
The output checksum of that function is then asserted to equal the value
recorded in every BLEEP gray run.json (``eb01186fd088f737``), so "identical
formula" is checked, not assumed.

Loaded by path (importlib) rather than ``import bleep.gray`` so it works no
matter whether bleep/ uses flat or package imports.

Batching
--------
``apply_gray_uint8`` accepts one (H, W, 3) array. A HisToGene item is a whole
section, (n, 112, 112, 3). Because the transform is strictly per pixel, the
section is reshaped to (n*112, 112, 3), passed through the canonical function
ONCE, and reshaped back. That is bit-identical to looping over patches
(``preflight_gray`` asserts this on a real cached section) and ~n times faster.

Where it is applied
-------------------
In ``histogene/dataset.py``, right after the uint8 patch is read from the
112x112 memmap cache and BEFORE the repo-order transpose / float cast / flatten.
The disk cache is never touched. Output stays 3 channels (luma replicated), so
``Linear(37632 -> 1024)`` is unchanged, and the 0-255 scale is unchanged
(HisToGene does not divide by 255).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from . import config as C

GRAY_SOURCE = C.REPO_ROOT / "bleep" / "gray.py"

# Output checksum recorded by the BLEEP colour-ablation runs. Override ONLY for
# local testing against a stand-in gray.py; never on the pod.
EXPECTED_CHECKSUM = os.environ.get("HTG_GRAY_EXPECTED", "eb01186fd088f737")

_MOD_NAME = "st_benching_bleep_gray"


def _load():
    if not GRAY_SOURCE.exists():
        raise FileNotFoundError(
            f"canonical grayscale module not found at {GRAY_SOURCE}. "
            "git pull st_benching - bleep/gray.py must be present.")
    if _MOD_NAME in sys.modules:
        return sys.modules[_MOD_NAME]
    spec = importlib.util.spec_from_file_location(_MOD_NAME, GRAY_SOURCE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod          # so inspect.getsource() can resolve it
    spec.loader.exec_module(mod)
    return mod


_g = _load()
apply_gray_uint8 = _g.apply_gray_uint8
gray_checksum = _g.gray_checksum
gray_fingerprint = _g.gray_fingerprint
_bleep_provenance = _g.gray_provenance
GRAY_MODES = _g.GRAY_MODES            # ("none", "query", "all")


def assert_canonical() -> str:
    """Raise unless the loaded transform produces the recorded checksum."""
    got = gray_checksum()
    if got != EXPECTED_CHECKSUM:
        raise RuntimeError(
            f"bleep/gray.py output checksum is {got}, expected {EXPECTED_CHECKSUM}. "
            "The transform differs from the one used by the existing perturbation "
            "control, so HisToGene numbers would not be comparable. Stop.")
    return got


def gray_section(patches: np.ndarray) -> np.ndarray:
    """(n, H, W, 3) uint8 -> (n, H, W, 3) uint8, luma replicated on 3 channels.

    One call to the canonical per-image function on a reshaped view; bit-identical
    to ``np.stack([apply_gray_uint8(p) for p in patches])``.
    """
    if not isinstance(patches, np.ndarray) or patches.dtype != np.uint8:
        raise TypeError(f"gray_section expects uint8 ndarray, got "
                        f"{type(patches).__name__} {getattr(patches, 'dtype', '')}")
    if patches.ndim != 4 or patches.shape[-1] != 3:
        raise ValueError(f"gray_section expects (n, H, W, 3), got {patches.shape}")
    n, h, w, c = patches.shape
    if n == 0:
        return patches.copy()
    flat = np.ascontiguousarray(patches).reshape(n * h, w, c)
    out = apply_gray_uint8(flat)
    return out.reshape(n, h, w, c)


def channel_spread_max(flat_patches: np.ndarray, patch_size: int = C.PATCH_SIZE) -> float:
    """Max over spots/pixels of (max channel - min channel) for a dataset item.

    Works on the flattened float tensor the model sees, (n, 3*P*P). The repo-order
    transpose keeps channels last, so reshaping to (n, P, P, 3) is valid.
    0 means every pixel has R == G == B, i.e. grayscale reached the model input.
    """
    a = np.asarray(flat_patches).reshape(-1, patch_size, patch_size, 3)
    return float((a.max(axis=-1) - a.min(axis=-1)).max()) if a.size else 0.0


def gray_provenance(mode: str) -> dict:
    """The BLEEP provenance block plus where/how it was applied in HisToGene."""
    if mode not in GRAY_MODES:
        raise ValueError(f"mode must be one of {GRAY_MODES}, got {mode!r}")
    d = dict(_bleep_provenance(mode))
    d.update({
        "gray_source": str(GRAY_SOURCE.relative_to(C.REPO_ROOT)),
        "gray_expected_checksum": EXPECTED_CHECKSUM,
        "gray_applied_where": ("histogene/dataset.py __getitem__: uint8 (n,112,112,3) "
                               "right after the memmap cache read, before transpose, "
                               "float cast and flatten; disk cache untouched; "
                               "3 channels kept; 0-255 scale kept"),
    })
    return d
