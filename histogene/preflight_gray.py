"""Pre-flight for the HisToGene colour ablation. Run before any training.

    python -m histogene.preflight_gray            # checks only (~1 min)
    python -m histogene.preflight_gray --forward  # + one GPU forward pass

Checks
  1. TRANSFORM   bleep/gray.py loads, output checksum == eb01186fd088f737,
                 pure red 200 -> 60 (0.299*200 = 59.8 -> 60)
  2. BATCHING    section-at-once gray == per-patch loop, bit-exact, on a real
                 cached section
  3. DATA PATH   HER2STSections(gray=True): R==G==B at the model input,
                 equals canonical-gray(cache) exactly, shape (n, 37632),
                 0-255 scale; gray=False still has colour and is unchanged
  4. CACHE       112x112 patch cache files not modified (size + mtime)
  5. MODEL       Linear in_features == 37632; forward on a gray item (--forward)
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import cache, config as C, her2st
from .dataset import HER2STSections
from .gray_bridge import (EXPECTED_CHECKSUM, GRAY_SOURCE, apply_gray_uint8,
                          channel_spread_max, gray_checksum, gray_fingerprint,
                          gray_section)

ok = True


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"[{'OK ' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""),
          flush=True)


def cache_snapshot():
    d = cache.patch_dir()
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(d.glob("*.npy"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--n_sections", type=int, default=3)
    args = ap.parse_args()

    print("=" * 70 + "\n1. TRANSFORM\n" + "=" * 70)
    check("bleep/gray.py present", GRAY_SOURCE.exists(), str(GRAY_SOURCE))
    got = gray_checksum()
    check("output checksum", got == EXPECTED_CHECKSUM,
          f"{got} (expect {EXPECTED_CHECKSUM})")
    print(f"       fingerprint {gray_fingerprint()}")
    red = np.zeros((2, 2, 3), np.uint8)
    red[..., 0] = 200
    v = int(apply_gray_uint8(red)[0, 0, 0])
    check("pure red 200 -> 60", v == 60, f"got {v}")
    rgb = np.array([[[10, 200, 30]]], np.uint8)
    exp = int(np.rint(0.299 * 10 + 0.587 * 200 + 0.114 * 30))
    check("hand-computed pixel", int(apply_gray_uint8(rgb)[0, 0, 0]) == exp,
          f"0.299*10+0.587*200+0.114*30 = {exp}")
    if not ok:
        sys.exit("transform is not the canonical one - stop")

    snap0 = cache_snapshot()
    names = her2st.section_names()
    panel = her2st.load_panel()
    check("patch cache present", len(snap0) >= len(names),
          f"{len(snap0)} files for {len(names)} sections in {cache.patch_dir()}")
    if not ok:
        sys.exit("run  python -m histogene.build_cache  first")

    print("\n" + "=" * 70 + "\n2. BATCHING (section-at-once vs per-patch)\n" + "=" * 70)
    raw = np.asarray(cache.load_patches(names[0]))
    t0 = time.time()
    a = gray_section(raw)
    t_batch = time.time() - t0
    t0 = time.time()
    b = np.stack([apply_gray_uint8(np.ascontiguousarray(p)) for p in raw])
    t_loop = time.time() - t0
    check("bit-identical", a.dtype == np.uint8 and np.array_equal(a, b),
          f"{names[0]} {raw.shape}")
    print(f"       batched {t_batch*1e3:.0f} ms vs loop {t_loop*1e3:.0f} ms")
    check("input array untouched", raw.shape[-1] == 3 and
          float((raw.max(-1).astype(int) - raw.min(-1)).max()) > 0,
          "raw cache still colour")

    print("\n" + "=" * 70 + "\n3. DATA PATH (HER2STSections)\n" + "=" * 70)
    secs = names[:args.n_sections]
    ds_c = HER2STSections(secs, panel, train=True, gray=False)
    ds_g = HER2STSections(secs, panel, train=True, gray=True)
    for i, s in enumerate(secs):
        xc, pc, ec = ds_c[i]
        xg, pg, eg = ds_g[i]
        xc, xg = xc.numpy(), xg.numpy()
        sc, sg = channel_spread_max(xc), channel_spread_max(xg)
        ref = gray_section(np.asarray(cache.load_patches(s))).transpose(0, 2, 1, 3)
        ref = ref.reshape(ref.shape[0], -1).astype(np.float32)
        check(f"{s} shapes unchanged", xc.shape == xg.shape and xg.shape[1] == C.PATCH_DIM,
              f"{xg.shape}")
        check(f"{s} gray: R==G==B at model input", sg == 0, f"spread {sg:.0f}")
        check(f"{s} colour: still colour", sc > 0, f"spread {sc:.0f}")
        check(f"{s} gray == canonical(cache)", np.array_equal(xg, ref))
        check(f"{s} scale 0-255 kept", xg.max() > 1.5 and xg.max() <= 255,
              f"max {xg.max():.0f} dtype {xg.dtype}")
        check(f"{s} positions/targets identical",
              bool((pc == pg).all()) and bool((ec == eg).all()))
        diff = float(np.abs(xc - xg).mean())
        check(f"{s} gray differs from colour", diff > 0, f"mean |diff| {diff:.2f}")

    # repo-order colour path is the old behaviour: transpose of the raw cache
    old = np.asarray(cache.load_patches(secs[0])).transpose(0, 2, 1, 3)
    old = old.reshape(old.shape[0], -1).astype(np.float32)
    check("gray=False == previous dataset behaviour", np.array_equal(ds_c[0][0].numpy(), old))

    print("\n" + "=" * 70 + "\n4. CACHE UNTOUCHED\n" + "=" * 70)
    check("patch cache size/mtime unchanged", cache_snapshot() == snap0,
          f"{len(snap0)} files")

    print("\n" + "=" * 70 + "\n5. MODEL\n" + "=" * 70)
    import torch
    sys.path.insert(0, str(C.HISTOGENE_REPO))
    from vis_model import HisToGene
    model = HisToGene(patch_size=C.PATCH_SIZE, n_layers=C.N_LAYERS, n_genes=len(panel),
                      dim=C.DIM, learning_rate=C.LR, dropout=C.DROPOUT, n_pos=C.N_POS)
    check("Linear(37632 -> 1024) unchanged",
          model.patch_embedding.in_features == C.PATCH_DIM
          and model.patch_embedding.out_features == C.DIM,
          f"{model.patch_embedding}")
    if args.forward:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev).eval()
        xg, pg, _ = ds_g[0]
        with torch.no_grad():
            y = model(xg.unsqueeze(0).to(dev), pg.unsqueeze(0).to(dev))
        check("forward on gray item", tuple(y.shape) == (1, xg.shape[0], len(panel)),
              f"{tuple(y.shape)} on {dev}")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
