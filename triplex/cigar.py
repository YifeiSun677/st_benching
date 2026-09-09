"""
Frozen CIGAR ResNet18 feature extractor (512-d).

TRIPLEX's target branch runs this same encoder INSIDE the model (trainable, up
to the last conv block -> 7x7x512). The neighbour and global branches consume
*precomputed* 512-d pooled features. To keep target and context on the same
encoder (as in the paper), we extract global/neighbour features with this exact
CIGAR ResNet18 (avgpool -> 512), frozen.

Weights: ozanciga self-supervised-histopathology "tenpercent" checkpoint, the
same file the upstream model downloads.
"""
import os
import torch
import torch.nn as nn
import torchvision

from . import config


def _load_state(ckpt_path):
    # The CIGAR file is a Lightning checkpoint: its pickle references
    # pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint, and torch>=2.6
    # defaults to weights_only=True which refuses it. Ensure that global is
    # resolvable (real pl if installed, else a throwaway stub) and load full.
    from .triplex_import import _ensure_pytorch_lightning, _patch_torch_load_weights_only
    _ensure_pytorch_lightning()
    _patch_torch_load_weights_only()
    state = torch.load(ckpt_path, map_location="cpu")   # weights_only=False via patch
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    for k in list(sd.keys()):
        sd[k.replace("model.", "").replace("resnet.", "")] = sd.pop(k)
    return sd


def build_cigar_encoder(ckpt_path=None, device="cuda"):
    """ResNet18 with CIGAR weights, fc removed -> outputs 512-d pooled features."""
    ckpt_path = ckpt_path or config.CIGAR_CKPT
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"CIGAR checkpoint missing at {ckpt_path}. Fetch it once with:\n"
            f"  mkdir -p {os.path.dirname(ckpt_path)} && "
            f"wget -O {ckpt_path} {config.CIGAR_URL}")

    resnet = torchvision.models.resnet18(weights=None)
    sd = _load_state(ckpt_path)
    md = resnet.state_dict()
    sd = {k: v for k, v in sd.items() if k in md}
    if not sd:
        raise RuntimeError("No CIGAR weights matched ResNet18 -- wrong checkpoint?")
    md.update(sd)
    resnet.load_state_dict(md)
    resnet.fc = nn.Identity()            # avgpool -> 512-d feature
    resnet.eval().to(device)
    for p in resnet.parameters():
        p.requires_grad_(False)
    return resnet


# ImageNet normalisation, matching the upstream target-branch transform.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def encode_patches(encoder, patches_uint8, device="cuda", batch=256):
    """
    patches_uint8 : (n, 224, 224, 3) uint8  ->  (n, 512) float32 numpy
    """
    import numpy as np
    feats = []
    mean = _MEAN.to(device)
    std = _STD.to(device)
    for i in range(0, len(patches_uint8), batch):
        chunk = patches_uint8[i:i + batch]
        x = torch.from_numpy(np.ascontiguousarray(chunk)).to(device).float() / 255.0
        x = x.permute(0, 3, 1, 2)        # nhwc -> nchw
        x = (x - mean) / std
        f = encoder(x)                   # (b, 512)
        feats.append(f.cpu())
    return torch.cat(feats, 0).numpy()
