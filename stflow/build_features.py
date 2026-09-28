"""
stflow_port/build_features.py -- one-off: crop every her2st spot, embed with frozen UNI,
and cache per-section arrays that train.py consumes.

STFlow never back-propagates into the image encoder: upstream runs
app/hest/benchmark.py once to write UNI embeddings to <embed_dataroot>/.../fp32/<id>.h5,
and app/flow/train.py only ever reads those. This script is the her2st equivalent.

Output: $STFLOW_CACHE/<tag>/<SEC>.npz with
    features  float32 [N, 1024]   UNI CLS embedding (fp32, eval mode)
    labels    float32 [N, 833]    log1p(raw counts), panel order, zero-filled
    raw       float32 [N, 833]    raw counts (same order) - kept for later re-normalisation
    coords    float32 [N, 2]      (pixel_x, pixel_y) full-res pixels  -> model coords (as HEST)
    array_xy  int32   [N, 2]      (x, y) ST array coordinates
    spot_id   str     [N]         'XxY'
    genes     str     [833]
plus manifest.json and qc/*.png.

Patch modes
  hest112    (default) crop 112 um of tissue (= round(112 / um_per_px) px, per section)
             and resize to 224 px: HEST-bench / UNI 20x (0.5 um/px) convention that STFlow
             was trained and published with.
  native224  crop 224 native pixels, no resize: the same field of view as the shared
             224x224 her2st_cache used by ST-Net/BLEEP (sensitivity analysis only).
  --grayscale  convert each crop to L then back to RGB before UNI (perturbation control).

Usage
  python stflow_port/build_features.py                       # hest112, all 36 sections
  python stflow_port/build_features.py --patch_mode native224
  python stflow_port/build_features.py --grayscale
  python stflow_port/build_features.py --sections A1 B1 --tag uni_v1_test   # quick test
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import her2st_io as H

Image.MAX_IMAGE_PIXELS = None   # her2st JPEGs are ~92 Mpx, above PIL's bomb threshold


def default_tag(patch_mode, grayscale):
    return f"uni_v1_{patch_mode}" + ("_gray" if grayscale else "")


def load_uni(device):
    import timm
    import torch
    if not os.path.isfile(C.UNI_CKPT):
        raise FileNotFoundError(f"UNI weights not found at {C.UNI_CKPT} (see RUNBOOK step 4)")
    model = timm.create_model(**C.UNI_TIMM_KWARGS)          # pretrained=False
    sd = torch.load(C.UNI_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)   # raises on any key mismatch
    model.eval().to(device)
    return model


def crop(img, cx, cy, size):
    """Square crop of `size` px centred on (cx=col, cy=row); white-padded at borders."""
    h, w = img.shape[:2]
    r0 = int(round(cy)) - size // 2
    c0 = int(round(cx)) - size // 2
    out = np.full((size, size, 3), 255, dtype=np.uint8)
    rs, re_ = max(r0, 0), min(r0 + size, h)
    cs, ce = max(c0, 0), min(c0 + size, w)
    if rs < re_ and cs < ce:
        out[rs - r0:re_ - r0, cs - c0:ce - c0] = img[rs:re_, cs:ce]
    return out


def to_uni_input(patch_u8, grayscale):
    """Upstream eval transform: Resize(224) -> ToTensor -> Normalize(ImageNet).
    torchvision's Resize on a PIL image delegates to PIL BILINEAR, so this is identical."""
    im = Image.fromarray(patch_u8)
    if grayscale:
        im = im.convert("L").convert("RGB")
    if im.size != (C.UNI_INPUT_PX, C.UNI_INPUT_PX):
        im = im.resize((C.UNI_INPUT_PX, C.UNI_INPUT_PX), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array(C.IMAGENET_MEAN, np.float32)) / np.array(C.IMAGENET_STD, np.float32)
    return a.transpose(2, 0, 1)


def embed(model, arrs, device, batch):
    import torch
    feats = []
    with torch.inference_mode():
        for i in range(0, len(arrs), batch):
            x = torch.from_numpy(np.stack(arrs[i:i + batch])).to(device)
            feats.append(model(x).float().cpu().numpy())
    return np.concatenate(feats, 0)


def qc_images(section, img, spots, crop_px, qc_dir, patches):
    os.makedirs(qc_dir, exist_ok=True)
    h, w = img.shape[:2]
    scale = 1500 / max(h, w)
    th = Image.fromarray(img).resize((int(w * scale), int(h * scale)))
    d = ImageDraw.Draw(th)
    for px, py in zip(spots["pixel_x"], spots["pixel_y"]):
        r = crop_px / 2 * scale
        d.rectangle([px * scale - r, py * scale - r, px * scale + r, py * scale + r],
                    outline=(0, 255, 0), width=2)
    th.save(os.path.join(qc_dir, f"{section}_overlay.png"))
    k = min(16, len(patches))
    idx = np.linspace(0, len(patches) - 1, k).astype(int)
    sheet = Image.new("RGB", (4 * 230, 4 * 245), "white")
    d2 = ImageDraw.Draw(sheet)
    for j, i in enumerate(idx):
        p = Image.fromarray(patches[i]).resize((224, 224))
        sheet.paste(p, ((j % 4) * 230, (j // 4) * 245))
        d2.text(((j % 4) * 230 + 2, (j // 4) * 245 + 226), spots.index[i], fill=(0, 0, 0))
    sheet.save(os.path.join(qc_dir, f"{section}_patches.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_mode", choices=["hest112", "native224"], default="hest112")
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--sections", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or default_tag(args.patch_mode, args.grayscale)
    out = C.cache_dir(tag)
    os.makedirs(out, exist_ok=True)
    genes = H.load_panel()
    assert len(genes) == len(set(genes)), "panel has duplicate genes"
    sections = args.sections or H.list_sections()
    print(f"[build_features] tag={tag} mode={args.patch_mode} gray={args.grayscale} "
          f"sections={len(sections)} genes={len(genes)} device={device}")

    model = load_uni(device)
    man_path = os.path.join(out, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {"sections": {}}
    manifest.update(tag=tag, patch_mode=args.patch_mode, grayscale=args.grayscale,
                    encoder="uni_v1_official", uni_ckpt=C.UNI_CKPT,
                    uni_ckpt_bytes=os.path.getsize(C.UNI_CKPT), precision="fp32",
                    target="log1p(raw counts), zero-filled", panel=C.PANEL,
                    genes=genes, her2st_root=C.HER2ST_ROOT)
    t_all = time.time()
    qc_done = set()
    for sec in sections:
        f_out = os.path.join(out, f"{sec}.npz")
        if os.path.exists(f_out) and not args.overwrite:
            print(f"  {sec}: exists, skip")
            continue
        t0 = time.time()
        counts, spots = H.load_section(sec)
        umpp, sx, sy = H.estimate_um_per_px(spots)
        crop_px = C.UNI_INPUT_PX if args.patch_mode == "native224" else int(round(C.HEST_PATCH_UM / umpp))
        img = np.asarray(Image.open(H.image_path(sec)).convert("RGB"))
        t_img = time.time() - t0
        patches = [crop(img, px, py, crop_px) for px, py in zip(spots["pixel_x"], spots["pixel_y"])]
        arrs = [to_uni_input(p, args.grayscale) for p in patches]
        t1 = time.time()
        feats = embed(model, arrs, device, args.batch)
        t_uni = time.time() - t1
        labels, raw, n_missing = H.compute_targets(counts, genes)
        assert np.isfinite(feats).all(), f"{sec}: non-finite UNI features"
        np.savez(f_out,
                 features=feats.astype(np.float32), labels=labels, raw=raw,
                 coords=spots[["pixel_x", "pixel_y"]].to_numpy(np.float32),
                 array_xy=spots[["x", "y"]].to_numpy(np.int32),
                 spot_id=np.array(spots.index, dtype=str), genes=np.array(genes, dtype=str),
                 section=sec, um_per_px=umpp, crop_px=crop_px)
        manifest["sections"][sec] = dict(n_spots=int(len(spots)), um_per_px=float(umpp),
                                         px_per_unit_x=float(sx), px_per_unit_y=float(sy),
                                         crop_px=int(crop_px), image_hw=list(img.shape[:2]),
                                         n_missing_genes=n_missing,
                                         sec_total=round(time.time() - t0, 1),
                                         sec_image=round(t_img, 1), sec_uni=round(t_uni, 1))
        if sec[0] not in qc_done:
            qc_images(sec, img, spots, crop_px, os.path.join(out, "qc"), patches)
            qc_done.add(sec[0])
        print(f"  {sec}: n={len(spots):4d} um/px={umpp:.3f} crop={crop_px}px "
              f"missing_genes={n_missing:3d}  image {t_img:.1f}s  uni {t_uni:.1f}s")
        json.dump(manifest, open(man_path, "w"), indent=2)
        del img, patches, arrs

    n = sum(v["n_spots"] for v in manifest["sections"].values())
    manifest["n_spots_total"] = n
    json.dump(manifest, open(man_path, "w"), indent=2)
    print(f"[build_features] done: {len(manifest['sections'])} sections, {n} spots, "
          f"{time.time() - t_all:.0f}s -> {out}")
    if len(manifest["sections"]) == C.EXPECTED_N_SECTIONS and n != C.EXPECTED_N_SPOTS:
        print(f"WARNING: {n} spots != benchmark's {C.EXPECTED_N_SPOTS}; check before training")


if __name__ == "__main__":
    main()
