"""
Load the REAL upstream TRIPLEX model (model/TRIPLEX/TRIPLEX.py + module.py) from
a read-only clone, without needing flash-attn, HEST, CLAM or MinkowskiEngine.

Two things upstream fights us on:

1. module.py does an *unconditional* top-level
       from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
   so the import fails if flash-attn isn't installed. We inject a stub module
   into sys.modules first. The stub's functions raise if ever called -- they
   never are, because we force flash_attn=False everywhere (see below).

2. GlobalEncoder defaults to flash_attn=True. The non-flash attention path is a
   plain exact softmax attention (mathematically identical), and crucially the
   masked / attention-bias paths the neighbour encoder relies on ONLY exist in
   the non-flash branch. So after building the model we recursively set
   flash_attn=False on every attention submodule -> deterministic, exact, and
   no CUDA flash kernel required.

We also avoid executing the repo's heavy model/__init__.py (which drags in the
Lightning ModelInterface and its deps) by registering lightweight package
objects for `model` and `model.TRIPLEX` pointing at the clone's source dirs.
"""
import os
import sys
import types
import importlib

from . import config


def _inject_flash_attn_stub():
    if "flash_attn" in sys.modules:
        return
    stub = types.ModuleType("flash_attn")

    def _unavailable(*_a, **_k):
        raise RuntimeError(
            "flash_attn was called but this port forces flash_attn=False. "
            "This should never happen; check disable_flash_attn().")

    stub.flash_attn_qkvpacked_func = _unavailable
    stub.flash_attn_func = _unavailable
    sys.modules["flash_attn"] = stub


def _ensure_pytorch_lightning():
    """TRIPLEX.py does a top-level `import pytorch_lightning as pl`. The model
    itself never uses pl (it's a plain nn.Module) -- BUT the CIGAR checkpoint is
    a Lightning checkpoint whose pickle references
    pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint. So:

      * If the real package is installed, use it (do NOT shadow it) -- it's what
        lets torch.load reconstruct that checkpoint. This is the simplest setup.
      * Only if pl is genuinely absent do we inject a stub deep enough to (a)
        satisfy the dead top-level import AND (b) let torch.load deserialize the
        checkpoint's ModelCheckpoint global as a throwaway dummy. That path lets
        you run with a *sanitized* (tensors-only) CIGAR ckpt and no pl at all.
    """
    try:
        import pytorch_lightning  # noqa: F401  (real package present -> use it)
        return
    except Exception:
        pass

    pl = types.ModuleType("pytorch_lightning")
    pl.LightningModule = object
    pl.LightningDataModule = object
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    mc = types.ModuleType("pytorch_lightning.callbacks.model_checkpoint")

    class _DummyCallback:
        """Reconstructable placeholder: pickle does __new__ + set __dict__."""
        def __init__(self, *a, **k):
            pass
        def __setstate__(self, state):
            try:
                self.__dict__.update(state)
            except Exception:
                pass

    mc.ModelCheckpoint = _DummyCallback
    callbacks.ModelCheckpoint = _DummyCallback
    callbacks.model_checkpoint = mc
    pl.callbacks = callbacks
    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    sys.modules["pytorch_lightning.callbacks.model_checkpoint"] = mc


def _patch_torch_load_weights_only():
    """Upstream TRIPLEX.py line ~45 calls torch.load(ckpt_path) with no
    weights_only, and it's in the read-only clone so we can't edit it. Under
    torch >= 2.6 the default flipped to weights_only=True, which refuses the
    checkpoint's ModelCheckpoint global. The CIGAR ckpt is a trusted release
    (ozanciga tenpercent), so default weights_only=False when the caller didn't
    specify it. Covers both upstream's target-encoder load and our cigar.py.
    Idempotent; a no-op on torch < 2.6 where the kwarg/behaviour differ."""
    import torch
    if getattr(torch.load, "_triplex_patched", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._triplex_patched = True
    _load._orig = _orig
    torch.load = _load


def _register_lean_packages(src_dir):
    """Make `model.TRIPLEX.*` importable without running model/__init__.py."""
    for pkg_name, pkg_path in [
        ("model", os.path.join(src_dir, "model")),
        ("model.TRIPLEX", os.path.join(src_dir, "model", "TRIPLEX")),
    ]:
        if pkg_name in sys.modules:
            continue
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [pkg_path]          # so submodules resolve via this dir
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg


def disable_flash_attn(model):
    """Force every attention submodule onto the exact (non-flash) path."""
    n = 0
    for m in model.modules():
        if hasattr(m, "flash_attn"):
            m.flash_attn = False
            n += 1
    return n


def load_triplex_class(repo=None):
    """Return the upstream TRIPLEX nn.Module class."""
    repo = repo or config.TRIPLEX_REPO
    src_dir = os.path.join(repo, "src")
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"TRIPLEX clone not found at {repo} (expected {src_dir}). "
            "Clone it read-only first; see the runbook.")

    _inject_flash_attn_stub()
    _ensure_pytorch_lightning()
    _patch_torch_load_weights_only()
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    _register_lean_packages(src_dir)

    # import module.py first so TRIPLEX.py's `from model.TRIPLEX.module import`
    # resolves against our registered package, then the model file itself.
    importlib.import_module("model.TRIPLEX.module")
    tri_mod = importlib.import_module("model.TRIPLEX.TRIPLEX")
    return tri_mod.TRIPLEX


def build_model(model_kwargs=None, weights_hint=None):
    """
    Construct TRIPLEX and neutralise flash-attn.

    The upstream __init__ downloads the CIGAR ResNet18 into ./weights/cigar
    relative to CWD. run.sh symlinks ./weights -> the canonical copy so this is
    a no-op after the first fetch. `weights_hint` is only used for a friendlier
    error message.
    """
    TRIPLEX = load_triplex_class()
    kwargs = dict(config.MODEL_KWARGS if model_kwargs is None else model_kwargs)
    model = TRIPLEX(**kwargs)
    n = disable_flash_attn(model)
    if n == 0:
        raise RuntimeError("disable_flash_attn touched 0 modules -- model layout changed?")
    return model
