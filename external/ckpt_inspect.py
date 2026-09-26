#!/usr/bin/env python
"""Stage 0 -- print what is inside checkpoint files, plus the key fields of the
run.json next to each, so a wrong-config checkpoint is caught before inference.

usage:
  python external/ckpt_inspect.py '/workspace/runs/bleep_lopo_e10/*/last.pt'
  python external/ckpt_inspect.py --audit            # every model, main-table weights
"""
import argparse
import glob
import json
import os
import sys

import torch

AUDIT = {   # model -> (glob of the 8 main-table weight files, expected epoch/budget)
    "bleep":      ("/workspace/runs/bleep_lopo_e10/[A-H]/last.pt", 10),
    "stnet":      ("/workspace/ST-Net/output/densenet121_224/top_833/[A-H]_checkpoints/epoch_25.pt", 25),
    "deeppt_ae":  ("/workspace/deeppt/results/deeppt_833_raw/ckpt/[A-H]_ae.pt", None),
    "deeppt_mlp": ("/workspace/deeppt/results/deeppt_833_raw/ckpt/[A-H]_mlp.pt", "early-stopped final"),
    "histogene":  ("/workspace/runs/histogene_lopo_833_ckpt/fold0[0-7]_[A-H]/last.ckpt", 100),
    "hist2st":    ("/workspace/runs/hist2st_lopo_833/fold0[0-7]_[A-H]/model.pt", 350),
    "path2space": ("/workspace/p2s_ckpt/path2space_lopo_833_ckpt/fold_[A-H]/ik_0.pt", 200),
    "triplex":    ("/workspace/triplex_ckpt/triplex_lopo_833_e20_ckpt/fold_[A-H]/final.pt", 20),
    "stflow":     ("/workspace/runs/stflow_lopo_833_normtarget_e20/fold0[0-7]_[A-H]/last.pth", 20),
}
RUN_KEYS = ("epoch", "lr", "learning_rate", "panel", "seed", "test", "fold", "budget", "tag", "norm")


def summarize(obj, depth=0, max_depth=2):
    pad = "  " * (depth + 1)
    if isinstance(obj, torch.nn.Module):
        print(f"{pad}<nn.Module {type(obj).__name__}: {sum(p.numel() for p in obj.parameters()):,} params>")
    elif isinstance(obj, dict):
        tens = [v for v in obj.values() if torch.is_tensor(v)]
        if obj and len(tens) == len(obj):
            print(f"{pad}<state_dict: {len(tens)} tensors, {sum(t.numel() for t in tens):,} params; "
                  f"first key {next(iter(obj))!r}>")
            return
        for k, v in obj.items():
            if torch.is_tensor(v):
                print(f"{pad}{k}: tensor{tuple(v.shape)}")
            elif isinstance(v, (int, float, str, bool)) or v is None:
                print(f"{pad}{k}: {v!r}")
            elif isinstance(v, (dict, list, tuple)) and depth < max_depth:
                print(f"{pad}{k}: {type(v).__name__}[{len(v)}]")
                summarize(v, depth + 1, max_depth)
            else:
                print(f"{pad}{k}: {type(v).__name__}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{type(obj).__name__}[{len(obj)}] of {sorted({type(x).__name__ for x in obj})}")
        if obj and depth < max_depth:
            summarize(obj[0], depth + 1, max_depth)
    else:
        print(f"{pad}{type(obj).__name__}")


def run_json_fields(path):
    rj = os.path.join(os.path.dirname(path), "run.json")
    if not os.path.exists(rj):
        return None
    d = json.load(open(rj))
    flat = {}

    def walk(x, pre=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{pre}{k}.")
        elif any(s in pre.lower() for s in RUN_KEYS):
            flat[pre[:-1]] = x
    walk(d)
    return flat


def inspect(path):
    print(f"== {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        summarize(obj)
    except Exception as e:                       # pickled classes not importable, etc.
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
    rj = run_json_fields(path)
    if rj:
        print("  run.json:", json.dumps(rj, default=str)[:600])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--audit", action="store_true", help="count + inspect one file per model")
    ap.add_argument("--pythonpath", action="append", default=[],
                    help="dirs to put on sys.path if a checkpoint pickles model classes")
    a = ap.parse_args()
    sys.path[:0] = a.pythonpath
    if a.audit:
        for model, (pat, budget) in AUDIT.items():
            files = sorted(f for f in glob.glob(pat, recursive=True) if os.path.isfile(f))
            print(f"\n##### {model}: {len(files)} files matched (expect 8)  main-table budget: {budget}")
            for f in files:
                print("   ", f)
            if files:
                inspect(files[0])
        n_ik = len(glob.glob("/workspace/p2s_ckpt/path2space_lopo_833_ckpt/fold_[A-H]/ik_[0-6].pt"))
        print(f"\npath2space: {n_ik} ik files in total (expect 56 = 8 folds x 7)")
        return
    for pat in a.paths:
        for f in sorted(glob.glob(pat, recursive=True)):
            if os.path.isfile(f):
                inspect(f)


if __name__ == "__main__":
    main()
