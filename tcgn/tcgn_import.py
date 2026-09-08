"""
Import the upstream TCGN model UNCHANGED from the read-only clone.

The only two things that can trip the import on a fresh env:
  1. newer timm moved timm.models.layers -> timm.layers  (CMT_block imports the
     old path). We alias it back if needed so any timm >=0.6 works.
  2. model.py does `from memory_profiler import profile` at import time even
     though @profile is never used. memory_profiler is a tiny pure-python pkg;
     we require it, but also install a no-op shim if it is missing so the port
     never dies on that line.
"""
import os
import sys
import types
import importlib

from config import TCGN_REPO_DIR, CMT_WEIGHTS


def _shim_timm():
    import timm  # noqa: F401
    try:
        importlib.import_module("timm.models.layers")
        return
    except Exception:
        pass
    try:
        layers = importlib.import_module("timm.layers")
        sys.modules["timm.models.layers"] = layers
    except Exception as e:
        raise ImportError(
            "timm is installed but neither timm.models.layers nor timm.layers "
            "could be imported: %r. Try `pip install timm==0.9.16`." % (e,)
        )


def _shim_memory_profiler():
    try:
        importlib.import_module("memory_profiler")
        return
    except Exception:
        mod = types.ModuleType("memory_profiler")
        mod.profile = lambda f=None, *a, **k: (f if callable(f) else (lambda g: g))
        sys.modules["memory_profiler"] = mod


def load_tcgn(num_classes, load_cmt=True, device="cuda"):
    """Return a TCGN(num_classes) with CMT-Tiny ImageNet weights (strict=False)."""
    _shim_timm()
    _shim_memory_profiler()
    if TCGN_REPO_DIR not in sys.path:
        sys.path.insert(0, TCGN_REPO_DIR)     # so `from model import TCGN` resolves
    from model import TCGN                     # noqa: E402  (upstream file, unedited)

    import torch
    model = TCGN(num_classes=num_classes)
    if load_cmt:
        if not os.path.exists(CMT_WEIGHTS):
            raise FileNotFoundError(
                "CMT weights not found at %s. Run:\n"
                "  cd %s/pretrained && bash download.sh\n"
                "(downloads cmt_tiny.pth from the CMT.pytorch release)."
                % (CMT_WEIGHTS, TCGN_REPO_DIR)
            )
        sd = torch.load(CMT_WEIGHTS, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # strict=False is expected: the classifier head + GNN blocks are not in
        # the CMT checkpoint. We print counts so a totally-wrong file is obvious.
        print("[tcgn_import] CMT loaded strict=False | missing=%d unexpected=%d"
              % (len(missing), len(unexpected)))
    return model.to(device)


if __name__ == "__main__":
    # CPU smoke test: instantiate, forward one fake batch, count params.
    import torch
    m = load_tcgn(num_classes=833, load_cmt=False, device="cpu")
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = m(x)
    print("params=%.3fM  out=%s (expect [2, 833])" % (n / 1e6, tuple(y.shape)))
