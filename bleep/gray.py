"""Canonical grayscale perturbation for the BLEEP colour ablation.

SINGLE SOURCE OF TRUTH.

Arm 2 (colour-trained -> grayscale test) and Arm 3 (grayscale-trained ->
grayscale test) MUST both obtain their grayscale patches by calling
`apply_gray_uint8` from this module. Never inline a second copy of the luma
formula anywhere else in the repo: the whole comparison is void if the two
arms see even slightly different pixels.

To make that enforceable rather than a promise, every run writes
`gray_checksum()` into its run.json, and `score_gray_paired.py` refuses to
compare two grayscale arms whose checksums differ.

Where this is applied
---------------------
On the uint8 HxWx3 patch as it comes out of the 224 patch cache, i.e. BEFORE
ToTensor and BEFORE ImageNet normalisation. The ImageNet mean/std are
per-channel and unequal, so normalisation is applied to the grayscale patch
exactly as it is to a colour one; the three channels go in equal and come out
unequal. That is deliberate and identical across arms.
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np

# ITU-R BT.601 luma coefficients. Same weights PIL uses for "RGB" -> "L"
# and the same ones torchvision's rgb_to_grayscale uses.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

GRAY_MODES = ("none", "query", "all")
# none  -> Arm 1: colour train, colour test (already finished)
# query -> Arm 2: colour train, grayscale applied ONLY to the held-out
#          section's query patches at inference. Reference/training spots stay
#          in colour, because the perturbation is of the test input.
# all   -> Arm 3: grayscale applied to every image patch the model ever sees,
#          during training and at inference.


def apply_gray_uint8(patch: np.ndarray) -> np.ndarray:
    """RGB uint8 (H, W, 3) -> luma replicated across 3 channels, uint8 (H, W, 3).

    Kept in uint8 in and uint8 out so the operation is exactly reproducible and
    can be dropped into the cache read path with no dtype surprises downstream.
    """
    if not isinstance(patch, np.ndarray):
        raise TypeError(f"apply_gray_uint8 expects np.ndarray, got {type(patch)!r}")
    if patch.dtype != np.uint8:
        raise TypeError(f"apply_gray_uint8 expects uint8, got {patch.dtype}")
    if patch.ndim != 3 or patch.shape[-1] != 3:
        raise ValueError(f"apply_gray_uint8 expects (H, W, 3), got {patch.shape}")

    # Explicit elementwise arithmetic in float64, not a matmul: a 3-term dot
    # product dispatched to BLAS can differ by one ulp between builds, which is
    # enough to flip a pixel that lands exactly on a .5 rounding boundary. This
    # form is bit-identical on every machine.
    p = patch.astype(np.float64)
    lum = _LUMA[0] * p[..., 0] + _LUMA[1] * p[..., 1] + _LUMA[2] * p[..., 2]
    lum = np.rint(lum).clip(0, 255).astype(np.uint8)
    return np.repeat(lum[:, :, None], 3, axis=2)


def maybe_gray(patch: np.ndarray, gray_mode: str, *, is_query: bool) -> np.ndarray:
    """Apply the grayscale transform according to the arm.

    `is_query=True` marks a patch belonging to the held-out section being
    predicted. Training patches and reference-bank patches are `is_query=False`.
    """
    if gray_mode not in GRAY_MODES:
        raise ValueError(f"gray_mode must be one of {GRAY_MODES}, got {gray_mode!r}")
    if gray_mode == "none":
        return patch
    if gray_mode == "all":
        return apply_gray_uint8(patch)
    # gray_mode == "query"
    return apply_gray_uint8(patch) if is_query else patch


def gray_any(patch, gray_mode: str, *, is_query: bool):
    """Same as maybe_gray, but accepts a PIL Image or a uint8 array.

    `her2st_dataset.Section.patch(i)` may hand back either depending on whether
    the patch cache or the JPEG path is in use, so this wrapper takes whatever
    comes and returns the same type. That keeps the dataset edit to one line
    and removes any chance of the two grayscale arms differing because one went
    through PIL and the other through numpy.
    """
    if gray_mode == "none":
        return patch
    if not (gray_mode == "all" or (gray_mode == "query" and is_query)):
        return patch

    if isinstance(patch, np.ndarray):
        return apply_gray_uint8(patch)

    # PIL Image. Convert via numpy rather than Image.convert("L") so the exact
    # same arithmetic is used on both paths.
    from PIL import Image
    arr = np.asarray(patch)
    if arr.ndim == 3 and arr.shape[-1] == 4:  # RGBA
        arr = arr[..., :3]
    return Image.fromarray(apply_gray_uint8(np.ascontiguousarray(arr)))


def gray_fingerprint() -> str:
    """Hash of the transform's source text. Catches edits to the code."""
    src = inspect.getsource(apply_gray_uint8) + repr(_LUMA.tolist())
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def gray_checksum() -> str:
    """Hash of the transform's OUTPUT on a fixed synthetic patch.

    This is the guarantee that matters. It is insensitive to comments and
    refactors but changes the moment a single output pixel changes, which is
    precisely the condition under which arms 2 and 3 stop being comparable.
    """
    rng = np.random.default_rng(0)
    probe = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    out = apply_gray_uint8(probe)
    return hashlib.sha256(out.tobytes()).hexdigest()[:16]


def gray_provenance(gray_mode: str) -> dict:
    """Block to merge into run.json for every arm, including the colour arm."""
    return {
        "gray_mode": gray_mode,
        "gray_checksum": gray_checksum(),
        "gray_fingerprint": gray_fingerprint(),
        "gray_luma": _LUMA.tolist(),
        "gray_applied_where": "uint8 patch from cache, before ToTensor/ImageNet norm",
    }


if __name__ == "__main__":
    print(f"gray_fingerprint : {gray_fingerprint()}")
    print(f"gray_checksum    : {gray_checksum()}")
    demo = np.zeros((4, 4, 3), dtype=np.uint8)
    demo[..., 0] = 200  # pure red
    print(f"pure red 200 -> {apply_gray_uint8(demo)[0, 0]}  (expect 60)")
