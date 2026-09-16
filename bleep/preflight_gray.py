#!/usr/bin/env python
"""Preflight for the BLEEP patient-B colour ablation.

Run this before the smoke test and again before the full run. It fails loudly
rather than letting a run start against a half-present cache or a missing
checkpoint, which on this project has cost more time than any single training
job.

Checks
  1. torch + CUDA visible, GPU name and free memory
  2. the 224 patch cache exists, is the expected shape/dtype, and is readable
  3. patient B has exactly 6 sections and every one of them is in the cache index
  4. the grayscale transform imports and reports its checksum
  5. a real patch round-trips through the transform and actually loses colour
     (channel spread drops to 0) while keeping structure (spatial SD preserved)
  6. for arm 2 only: the 6 colour checkpoints from the finished arm-1 run exist

Usage
  python -m bleep.preflight_gray --cache /workspace/her2st_cache
  python -m bleep.preflight_gray --cache /workspace/her2st_cache \
      --arm1-run /workspace/runs/bleep_patientB_833 --require-checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from bleep.gray import apply_gray_uint8, gray_checksum, gray_fingerprint
except ImportError:  # running as a loose script from inside bleep/
    from gray import apply_gray_uint8, gray_checksum, gray_fingerprint  # type: ignore

PATIENT_B_SECTIONS = ["B1", "B2", "B3", "B4", "B5", "B6"]

_ok = True


def check(label: str, condition: bool, detail: str = "") -> None:
    global _ok
    mark = "PASS" if condition else "FAIL"
    if not condition:
        _ok = False
    line = f"[{mark}] {label}"
    if detail:
        line += f"  --  {detail}"
    print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/her2st_cache",
                    help="directory holding the 224 patch cache memmap + index")
    ap.add_argument("--arm1-run", default="",
                    help="finished colour run dir (for the checkpoint check)")
    ap.add_argument("--require-checkpoints", action="store_true")
    ap.add_argument("--patch-file", default="",
                    help="override the memmap filename if it is not auto-found")
    args = ap.parse_args()

    print("=" * 72)
    print("BLEEP patient-B colour ablation -- preflight")
    print("=" * 72)

    # 1. torch / CUDA -----------------------------------------------------
    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "none"
        check("torch imports", True, torch.__version__)
        check("CUDA available", cuda, name)
        if cuda:
            free, total = torch.cuda.mem_get_info()
            print(f"       GPU memory free {free/2**30:.1f} GB / {total/2**30:.1f} GB")
    except Exception as exc:  # noqa: BLE001
        check("torch imports", False, repr(exc))

    # 2. patch cache ------------------------------------------------------
    cache_dir = Path(args.cache)
    check("patch cache dir exists", cache_dir.is_dir(), str(cache_dir))

    patch_path = None
    if args.patch_file:
        patch_path = Path(args.patch_file)
    elif cache_dir.is_dir():
        candidates = sorted(
            [p for p in cache_dir.iterdir()
             if p.suffix in (".npy", ".dat", ".mmap", ".bin")],
            key=lambda p: p.stat().st_size, reverse=True)
        patch_path = candidates[0] if candidates else None

    if patch_path is not None and patch_path.exists():
        size_gb = patch_path.stat().st_size / 2**30
        check("patch memmap present", True, f"{patch_path.name}  {size_gb:.2f} GB")
        # 13,620 spots x 224 x 224 x 3 uint8 = 2.05 GB
        expected_gb = 13620 * 224 * 224 * 3 / 2**30
        check("patch memmap size within 5% of 13,620 x 224 x 224 x 3 uint8",
              abs(size_gb - expected_gb) / expected_gb < 0.05,
              f"expected {expected_gb:.2f} GB")
        try:
            if patch_path.suffix == ".npy":
                arr = np.load(patch_path, mmap_mode="r")
            else:
                arr = np.memmap(patch_path, dtype=np.uint8, mode="r")
                arr = arr.reshape(-1, 224, 224, 3)
            check("patch memmap readable", True, f"shape {arr.shape} dtype {arr.dtype}")
            sample = np.asarray(arr[0])
        except Exception as exc:  # noqa: BLE001
            check("patch memmap readable", False, repr(exc))
            sample = None
    else:
        check("patch memmap present", False, "no candidate file found in cache dir")
        sample = None

    # 3. section index ----------------------------------------------------
    idx_candidates = []
    if cache_dir.is_dir():
        idx_candidates = [p for p in cache_dir.iterdir()
                          if p.suffix in (".json", ".csv", ".txt", ".tsv")]
    if idx_candidates:
        joined = " ".join(p.name for p in idx_candidates)
        check("cache index file(s) present", True, joined)
        blob = ""
        for p in idx_candidates:
            try:
                blob += p.read_text(errors="ignore")
            except Exception:  # noqa: BLE001
                pass
        missing = [s for s in PATIENT_B_SECTIONS if s not in blob]
        check("all 6 patient-B sections appear in the cache index",
              not missing, f"missing: {missing}" if missing else "B1..B6")
    else:
        check("cache index file(s) present", False,
              "expected a json/csv listing spot ids and sections")

    # 4. grayscale transform ---------------------------------------------
    print(f"       gray_fingerprint {gray_fingerprint()}")
    print(f"       gray_checksum    {gray_checksum()}")
    check("grayscale transform imports", True, "bleep.gray")

    # 5. behaviour on a real patch ---------------------------------------
    if sample is not None:
        g = apply_gray_uint8(sample)
        colour_spread = float(np.mean(np.ptp(sample.astype(np.float64), axis=2)))
        gray_spread = float(np.mean(np.ptp(g.astype(np.float64), axis=2)))
        check("grayscale removes channel spread", gray_spread == 0.0,
              f"colour mean channel range {colour_spread:.2f} -> gray {gray_spread:.2f}")
        sd_before = float(sample.astype(np.float64).mean(axis=2).std())
        sd_after = float(g[..., 0].astype(np.float64).std())
        keeps = sd_before > 0 and abs(sd_after - sd_before) / sd_before < 0.35
        check("grayscale keeps spatial structure", keeps,
              f"spatial SD {sd_before:.2f} -> {sd_after:.2f}")
        check("grayscale patch is not constant", sd_after > 1.0,
              "a flat patch here would mean the cache row is blank")

    # 6. arm-1 checkpoints -------------------------------------------------
    if args.require_checkpoints:
        run_dir = Path(args.arm1_run)
        check("arm-1 run dir exists", run_dir.is_dir(), str(run_dir))
        if run_dir.is_dir():
            ckpts = sorted(run_dir.rglob("*.pt")) + sorted(run_dir.rglob("*.pth"))
            check("6 colour checkpoints found (one per fold)", len(ckpts) >= 6,
                  f"{len(ckpts)} found")
            for c in ckpts[:8]:
                print(f"       {c.relative_to(run_dir)}  {c.stat().st_size/2**20:.0f} MB")
            runjsons = sorted(run_dir.rglob("run.json"))
            if runjsons:
                cfg = json.loads(runjsons[0].read_text())
                ep = cfg.get("epochs", cfg.get("config", {}).get("epochs", "?"))
                print(f"       arm-1 epochs recorded in run.json: {ep}  (must be 20)")
                for key in ("train_seconds", "elapsed_seconds", "timing", "wall_time"):
                    if key in cfg:
                        print(f"       arm-1 {key}: {cfg[key]}  <-- use this for the time estimate")

    print("=" * 72)
    print("PREFLIGHT OK" if _ok else "PREFLIGHT FAILED -- fix the above before running")
    print("=" * 72)
    return 0 if _ok else 1


if __name__ == "__main__":
    sys.exit(main())
