#!/usr/bin/env python
"""Quick inference-efficiency test: run each external driver UNCHANGED on one fold and
one section, and sample the whole process tree from outside.

No driver edits, no retraining -- it reuses the trained LOPO checkpoints. For every model:
    wall_s                end-to-end seconds (model load + preprocessing + predict + write)
    n_spots               rows in the written preds/<sec>.npz
    spots_per_s           n_spots / wall_s
    peak_gpu_mem_gb       nvidia-smi memory.used peak minus the idle baseline
    mean/max_gpu_util     nvidia-smi, 1 Hz
    peak_host_rss_gb      RSS summed over the driver and its children (dataloader workers)

    python external/bench_efficiency.py                        # all models, fold B, section I1
    python external/bench_efficiency.py --models stflow triplex --sections I1 J1 --repeats 2

The GPU must be otherwise idle; GPU memory is device-wide, not per process.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
MODELS = ["bleep", "deeppt", "hist2st", "histogene", "path2space", "stflow", "stnet", "triplex"]


def gpu_sample(dev: int):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={dev}", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, text=True)
        u, m = out.strip().split(",")
        return float(u), float(m) / 1024  # GiB
    except Exception:
        return None, None


def tree_rss(proc: psutil.Process) -> float:
    total = 0.0
    try:
        for p in [proc, *proc.children(recursive=True)]:
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except psutil.NoSuchProcess:
        pass
    return total / 1e9


def run_one(model: str, args, rep: int) -> dict:
    out = Path(args.out) / f"{model}_rep{rep}"
    cmd = [sys.executable, str(HERE / f"run_{model}.py"), "--folds", args.fold,
           "--sections", *args.sections, "--skip_roundtrip", "--out", str(out)]
    _, base_mem = gpu_sample(args.gpu)
    samples, stop = [], threading.Event()

    t0 = time.perf_counter()
    log = open(Path(args.out) / f"{model}_rep{rep}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    ps = psutil.Process(proc.pid)

    def loop():
        while not stop.is_set():
            u, m = gpu_sample(args.gpu)
            samples.append((u, m, tree_rss(ps)))
            stop.wait(args.interval)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    rc = proc.wait()
    wall = time.perf_counter() - t0
    stop.set()
    th.join()
    log.close()

    n_spots = 0
    for f in out.rglob("preds/*.npz"):
        n_spots += len(np.load(f, allow_pickle=True)["spot_id"])
    utils = [s[0] for s in samples if s[0] is not None]
    mems = [s[1] for s in samples if s[1] is not None]
    return {
        "model": model, "rep": rep, "returncode": rc, "fold": args.fold,
        "sections": " ".join(args.sections), "wall_s": round(wall, 2), "n_spots": n_spots,
        "spots_per_s": round(n_spots / wall, 1) if n_spots else None,
        "peak_gpu_mem_gb": round(max(mems) - (base_mem or 0), 3) if mems else None,
        "mean_gpu_util_pct": round(float(np.mean(utils)), 1) if utils else None,
        "max_gpu_util_pct": max(utils) if utils else None,
        "peak_host_rss_gb": round(max(s[2] for s in samples), 3) if samples else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--fold", default="B", help="held-out her2st patient whose weights are used")
    ap.add_argument("--sections", nargs="*", default=["I1"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--out", default="/workspace/runs/efficiency")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    rows = []
    for m in args.models:
        for r in range(args.repeats):
            print(f"[{m} rep{r}] running ...", flush=True)
            row = run_one(m, args, r)
            print(json.dumps(row), flush=True)
            rows.append(row)
            Path(args.out, "efficiency.json").write_text(json.dumps(rows, indent=2))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out) / "efficiency.csv", index=False)
    print(df.drop(columns=["fold", "sections"]).to_string(index=False))


if __name__ == "__main__":
    main()
