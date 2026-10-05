#!/usr/bin/env python
"""Count parameters of the 8 benchmarked models from their saved fold-A weights.

  trainable  = tensors in the saved checkpoint (what each run actually trained)
  frozen     = pretrained image encoder used only as a feature extractor (not in the checkpoint)

BatchNorm buffers (running_mean / running_var / num_batches_tracked) and optimizer state are
not parameters and are skipped.  Path2Space saves an ensemble: 7 ik files x N_IL MLPs per fold;
both one MLP and the whole ensemble are reported.

usage: cd /workspace/st_benching && python external/count_params.py [--fold A]
"""
import argparse
import glob
import os

import torch

BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def count(obj) -> int:
    """Sum numel over every parameter tensor in a (possibly nested) checkpoint."""
    if isinstance(obj, torch.Tensor):
        return obj.numel()
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):  # lightning
            obj = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):          # ST-Net, TRIPLEX resume
            obj = obj["model"]
        return sum(count(v) for k, v in obj.items()
                   if not (isinstance(k, str) and k.endswith(BUFFER_SUFFIXES))
                   and not (isinstance(k, str) and "optim" in k.lower()))
    if isinstance(obj, (list, tuple)):
        return sum(count(v) for v in obj)
    return 0


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def frozen_encoders():
    import torchvision
    tv = torchvision.models
    out = {}
    r50 = tv.resnet50(); r50.fc = torch.nn.Identity()
    out["ResNet-50 (ImageNet)"] = sum(p.numel() for p in r50.parameters())
    r18 = tv.resnet18(); r18.fc = torch.nn.Identity()
    out["ResNet-18 (CIGAR)"] = sum(p.numel() for p in r18.parameters())
    try:
        import timm
        uni = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0,
                                init_values=1e-5, dynamic_img_size=True)
        out["UNI ViT-L/16"] = sum(p.numel() for p in uni.parameters())
    except Exception as e:  # noqa: BLE001
        out["UNI ViT-L/16"] = f"timm unavailable ({e.__class__.__name__})"
    ctp = "/workspace/p2s_weights/ctranspath.pth"
    out["CTransPath"] = count(load(ctp)) if os.path.exists(ctp) else "weights not found"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="A")
    P = ap.parse_args().fold
    k = "ABCDEFGH".index(P)
    ckpts = {
        "ST-Net":     [f"/workspace/ST-Net/output/densenet121_224/top_833/{P}_checkpoints/epoch_25.pt"],
        "HisToGene":  [f"/workspace/runs/histogene_lopo_833_ckpt/fold{k:02d}_{P}/last.ckpt"],
        "Hist2ST":    [f"/workspace/runs/hist2st_lopo_833/fold{k:02d}_{P}/model.pt"],
        "BLEEP":      [f"/workspace/runs/bleep_lopo_e10/{P}/last.pt"],
        "DeepPT":     [f"/workspace/deeppt/results/deeppt_833_raw/ckpt/{P}_ae.pt",
                       f"/workspace/deeppt/results/deeppt_833_raw/ckpt/{P}_mlp.pt"],
        "TRIPLEX":    [f"/workspace/triplex_ckpt/triplex_lopo_833_e20_ckpt/fold_{P}/final.pt"],
        "STFlow":     [f"/workspace/runs/stflow_lopo_833_normtarget_e20/fold{k:02d}_{P}/last.pth"],
        "Path2Space": sorted(glob.glob(f"/workspace/p2s_ckpt/path2space_lopo_833_ckpt/fold_{P}/ik_*.pt")),
    }
    frozen_of = {"DeepPT": "ResNet-50 (ImageNet)", "TRIPLEX": "ResNet-18 (CIGAR)",
                 "STFlow": "UNI ViT-L/16", "Path2Space": "CTransPath"}
    enc = frozen_encoders()

    print(f"fold {P}\n{'model':<11} {'trainable':>14} {'frozen encoder':>34}")
    for name, files in ckpts.items():
        missing = [f for f in files if not os.path.exists(f)]
        if not files or missing:
            print(f"{name:<11} MISSING {missing or 'no files matched'}")
            continue
        n = sum(count(load(f)) for f in files)
        fz = frozen_of.get(name)
        fz_txt = f"{fz}: {enc[fz]:,}" if fz and isinstance(enc[fz], int) else (f"{fz}: {enc[fz]}" if fz else "-")
        line = f"{name:<11} {n:>14,} {fz_txt:>34}"
        if name == "Path2Space":
            one = count(load(files[0]))
            line += f"   ({len(files)} ik files; one ik file = {one:,})"
        print(line)


if __name__ == "__main__":
    main()
