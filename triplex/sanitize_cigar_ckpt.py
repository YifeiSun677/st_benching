"""
Optional, run ONCE. Rewrites the CIGAR Lightning checkpoint to a tensors-only
file ({'state_dict': ...}) so that afterwards it loads under torch's safe
default (weights_only=True) with NO pytorch_lightning dependency and no load
patch -- ideal since the cleaned file lives on the persistent /workspace volume
and survives pod rebuilds.

It reads the original by making the single offending global
(pytorch_lightning...ModelCheckpoint) resolvable as a throwaway stub, so this
step itself does not require pytorch_lightning to be installed.

  python -m triplex.sanitize_cigar_ckpt                 # in place (with backup)
  python -m triplex.sanitize_cigar_ckpt --out clean.ckpt
"""
import argparse
import os
import shutil
import torch

from . import config
from .triplex_import import _ensure_pytorch_lightning, _patch_torch_load_weights_only


def sanitize(src, out=None, keep_backup=True):
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    _ensure_pytorch_lightning()          # stub or real -> resolves ModelCheckpoint
    _patch_torch_load_weights_only()     # weights_only=False for the raw read
    state = torch.load(src, map_location="cpu")
    if not (isinstance(state, dict) and "state_dict" in state):
        raise RuntimeError("checkpoint has no 'state_dict' key -- unexpected format")
    sd = {k: v for k, v in state["state_dict"].items() if torch.is_tensor(v)}
    if not sd:
        raise RuntimeError("no tensors found in state_dict")

    out = out or src
    if out == src and keep_backup:
        bak = src + ".orig"
        if not os.path.exists(bak):
            shutil.copy2(src, bak)
            print(f"[sanitize] backed up original -> {bak}")
    torch.save({"state_dict": sd}, out)
    print(f"[sanitize] wrote tensors-only checkpoint ({len(sd)} tensors) -> {out}")

    # verify it now loads under the safe default, with pl absent
    chk = torch.load(out, map_location="cpu", weights_only=True)
    assert "state_dict" in chk and len(chk["state_dict"]) == len(sd)
    print("[sanitize] verified: loads under weights_only=True (no pl needed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=config.CIGAR_CKPT)
    ap.add_argument("--out", default=None, help="default: overwrite --src (keeps .orig backup)")
    args = ap.parse_args()
    sanitize(args.src, args.out)
