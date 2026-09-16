#!/usr/bin/env python
"""Locate the finished BLEEP patient-B colour run and decide what to do next.

Scans for directories containing preds.npz, groups them into runs, and reports
only what matters: how many folds, which sections, how many genes, whether
checkpoints exist. Then prints the branch and the exact next command.

  python -m bleep.find_arm1
  python -m bleep.find_arm1 --roots /workspace ~/Documents/science
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

CKPT_SUFFIXES = (".pt", ".pth", ".ckpt")


def scan(roots: list[Path], max_depth: int = 8) -> dict[Path, list[Path]]:
    """run_root -> [fold_dir, ...], where fold dirs hold a preds.npz."""
    fold_dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        root = root.resolve()
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            d = Path(dirpath)
            if len(d.parts) - base_depth > max_depth:
                dirnames[:] = []
                continue
            # never descend into data or cache trees
            dirnames[:] = [x for x in dirnames if x not in
                           {".git", "__pycache__", "node_modules",
                            "her2st_cache", "images", "tifs"}]
            if "preds.npz" in filenames:
                fold_dirs.append(d)

    runs: dict[Path, list[Path]] = defaultdict(list)
    for fd in fold_dirs:
        runs[fd.parent].append(fd)
    return {k: sorted(v) for k, v in sorted(runs.items())}


def describe(run_root: Path, folds: list[Path]) -> dict:
    sections: set[str] = set()
    n_genes = set()
    epochs = set()
    gray_modes = set()
    n_spots = 0

    for fd in folds:
        try:
            with np.load(fd / "preds.npz", allow_pickle=True) as z:
                qk = np.asarray(z["query_keys"]).astype(str)
                sections.update(k.split(":")[0] for k in qk)
                n_spots += len(qk)
                if "genes" in z.files:
                    n_genes.add(int(len(z["genes"])))
        except Exception as exc:  # noqa: BLE001
            sections.add(f"<unreadable:{type(exc).__name__}>")
        rj = fd / "run.json"
        if rj.exists():
            try:
                cfg = json.loads(rj.read_text())
                flat = cfg.get("config", cfg)
                for k in ("epochs", "n_epochs"):
                    if k in flat:
                        epochs.add(flat[k])
                if "gray_mode" in cfg:
                    gray_modes.add(cfg["gray_mode"])
            except Exception:  # noqa: BLE001
                pass

    ckpts = [p for p in run_root.rglob("*") if p.suffix in CKPT_SUFFIXES
             and p.is_file() and p.stat().st_size > 1_000_000]

    return {
        "run_root": run_root,
        "n_folds": len(folds),
        "sections": sorted(sections),
        "n_genes": sorted(n_genes),
        "n_spots": n_spots,
        "epochs": sorted(epochs),
        "gray_modes": sorted(gray_modes),
        "n_ckpt": len(ckpts),
        "ckpt_gb": sum(p.stat().st_size for p in ckpts) / 2**30,
        "example_ckpt": str(ckpts[0]) if ckpts else "",
    }


def is_patient_b(info: dict) -> bool:
    secs = info["sections"]
    return len(secs) == 6 and all(s.startswith("B") for s in secs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["/workspace"])
    args = ap.parse_args()

    roots = [Path(os.path.expanduser(r)) for r in args.roots]
    print(f"scanning: {', '.join(str(r) for r in roots)}\n")
    runs = scan(roots)

    if not runs:
        print("No preds.npz found anywhere under those roots.")
        print("=> BRANCH C: the finished run is not on this machine.")
        print("   Look on the Mac under ~/Documents/science/bleep_out/runs/")
        print("   and rerun with --roots ~/Documents/science")
        return 1

    infos = [describe(rr, fds) for rr, fds in runs.items()]

    print(f"{'folds':>6} {'genes':>6} {'spots':>7} {'ep':>4} {'ckpt':>5} "
          f"{'gray':>6}  sections            run")
    print("-" * 110)
    for i in infos:
        secs = ",".join(i["sections"])[:18]
        gray = ",".join(i["gray_modes"]) or "-"
        ep = ",".join(str(e) for e in i["epochs"]) or "?"
        print(f"{i['n_folds']:>6} {str(i['n_genes'])[1:-1]:>6} {i['n_spots']:>7} "
              f"{ep:>4} {i['n_ckpt']:>5} {gray:>6}  {secs:<18}  {i['run_root']}")

    # the arm-1 candidate: 6 B sections, and either no gray_mode recorded
    # (predates this drop) or gray_mode == none (the colour arm)
    cands = [i for i in infos if is_patient_b(i)
             and set(i["gray_modes"]) <= {"none"}]
    cands.sort(key=lambda i: (i["n_ckpt"] > 0, 833 in i["n_genes"]), reverse=True)

    print("\n" + "=" * 78)
    if not cands:
        print("No run covers exactly B1-B6 with no gray_mode recorded.")
        print("=> The colour control is not on this machine (BRANCH C),")
        print("   or its sections are named differently. Check the table above.")
        return 1

    best = cands[0]
    print(f"ARM-1 CANDIDATE: {best['run_root']}")
    print(f"  folds {best['n_folds']}  sections {','.join(best['sections'])}  "
          f"genes {best['n_genes']}  spots {best['n_spots']}  "
          f"epochs {best['epochs']}")
    print(f"  checkpoints: {best['n_ckpt']} ({best['ckpt_gb']:.1f} GB)")
    if len(cands) > 1:
        print(f"  ({len(cands) - 1} other patient-B candidate(s) listed above)")

    print("=" * 78)
    if best["n_ckpt"] >= best["n_folds"]:
        print("=> BRANCH A: predictions and checkpoints both present. Proceed.\n")
        print(f'  export ARM1_RUN="{best["run_root"]}"')
        print("  python -m bleep.preflight_gray --cache /workspace/her2st_cache \\")
        print(f'      --arm1-run "$ARM1_RUN" --require-checkpoints')
        print("  ./bleep/run_gray_patientB.sh smoke")
    else:
        print("=> BRANCH B: predictions present, checkpoints MISSING.")
        print("   Arm 2 is the colour model fed grayscale input, so it cannot")
        print("   run without weights. Retrain the colour arm at current HEAD")
        print("   with --save_ckpt, which also reproduces +0.366 as a check.\n")
        print(f'  export ARM1_RUN="{best["run_root"]}_repro"')
        print("  EPOCHS=20 ARM3_RUN=\"$ARM1_RUN\" \\")
        print("    GRAY_MODE=none ./bleep/run_gray_patientB.sh arm3   "
              "# edit --gray all -> none, --save_ckpt on")
        print("\n   Then score the reproduced run against the original preds")
        print("   before starting arms 2 and 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
