#!/usr/bin/env python
"""
Step 1 — spot patches -> frozen ResNet50 -> 2048-d features.

Replaces DeepPT's 11slide_processing/{1main_processing,collect_features}.py,
which tile a whole slide and drop background. You have spot coordinates, so
tissue detection is unnecessary and would only desynchronise your spot set
from the ST-Net / HisToGene runs.

ONE-OFF: the ResNet50 is frozen, so this does NOT repeat per LOPO fold.

Output (per section):
    features/<sec>.npy       float32 [n_spots, 2048]
    features/<sec>_spots.csv spot_id, x, y, pixel_x, pixel_y   (row-aligned)

Usage:
    python 01_extract_features.py \
        --her2st /workspace/her2st \
        --weights /workspace/DeepPT_original/ResNet50_IMAGENET1K_V2.pt \
        --out /workspace/deeppt/features_raw \
        --colornorm raw
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50
from tqdm import tqdm

import her2st_io as io

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_encoder(weights_path: str | None, device: str) -> nn.Module:
    """ResNet50 truncated after global average pool -> 2048-d."""
    net = resnet50(weights=None)
    if weights_path:
        obj = torch.load(weights_path, map_location="cpu", weights_only=False)
        sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
        sd = sd.get("state_dict", sd)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"[resnet50] loaded {weights_path}")
        print(f"           missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 5:
            raise RuntimeError("checkpoint does not match resnet50 — inspect keys")
    else:
        from torchvision.models import ResNet50_Weights
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        print("[resnet50] fell back to torchvision IMAGENET1K_V2")
    net.fc = nn.Identity()
    return net.eval().to(device)


def make_normalizer(mode: str, ref_patches: np.ndarray | None):
    if mode == "raw":
        return lambda p: p
    if mode != "macenko":
        raise ValueError(mode)
    import torchstain
    norm = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
    norm.fit(ref_patches.reshape(-1, ref_patches.shape[-2], 3)
             if ref_patches.ndim == 4 else ref_patches)

    def _apply(patches: np.ndarray) -> np.ndarray:
        out = np.empty_like(patches)
        for i, p in enumerate(patches):
            try:
                out[i] = norm.normalize(I=p, stains=False)[0].astype(np.uint8)
            except Exception:
                out[i] = p  # degenerate (near-white) patch — leave as-is
        return out
    return _apply


@torch.no_grad()
def encode(patches: np.ndarray, net: nn.Module, device: str, batch: int) -> np.ndarray:
    feats = []
    for i in range(0, len(patches), batch):
        chunk = patches[i:i + batch].astype(np.float32) / 255.0
        chunk = (chunk - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            f = net(t)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--her2st", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--colornorm", choices=["raw", "macenko"], default="raw")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--sections", nargs="*", default=None)
    ap.add_argument("--save-patches", action="store_true",
                    help="also cache uint8 patches (~2 GB for all 13,620 spots)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    net = build_encoder(args.weights, device)

    sections = args.sections or io.list_sections(args.her2st)
    print(f"[sections] {len(sections)}: {sections}")

    # Macenko reference: first section, fitted once and reused for all sections
    # (fitting per-section would defeat the purpose of removing batch effects).
    normalize = None
    total_spots = 0
    t0 = time.time()

    for sec in tqdm(sections, desc="sections"):
        fout = os.path.join(args.out, f"{sec}.npy")
        if os.path.exists(fout):
            total_spots += len(np.load(fout, mmap_mode="r"))
            continue

        pos = io.aligned_spots(args.her2st, sec)
        from PIL import Image
        img = Image.open(io.image_path(args.her2st, sec)).convert("RGB")
        patches = io.crop_patches(img, pos)
        img.close()

        if normalize is None:
            normalize = make_normalizer(args.colornorm, patches[:64])
        patches = normalize(patches)

        if args.save_patches:
            np.save(os.path.join(args.out, f"{sec}_patches.npy"), patches)

        feats = encode(patches, net, device, args.batch)
        np.save(fout, feats.astype(np.float32))
        pos.reset_index(names="spot_id").to_csv(
            os.path.join(args.out, f"{sec}_spots.csv"), index=False)
        total_spots += len(feats)

    print(f"[done] {total_spots} spots, {time.time() - t0:.0f}s -> {args.out}")
    print("       sanity: expect 13,620 spots over 36 sections")


if __name__ == "__main__":
    main()
