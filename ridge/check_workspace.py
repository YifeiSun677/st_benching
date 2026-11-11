"""
ridge/check_workspace.py -- READ-ONLY inventory of everything the linear baselines
need on the pod. Run it first; it changes nothing except writing its own report.

    python -m ridge.check_workspace
    python -m ridge.check_workspace --preds TRIPLEX=/workspace/triplex_results/triplex_lopo_833 \
                                    STFlow=/workspace/runs/stflow_lopo_833_hest112_e90

For each item it prints PASS / WARN / FAIL / MISSING and a hint. --preds is optional:
for each model run dir given, it reads one prediction file and reports which target
space (data.TARGETS) reproduces that model's stored truth, so the paired ridge run
can be scored in exactly the same space.

Report: $RIDGE_CACHE/workspace_check.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

from . import config, data, features

RESULTS = []


def report(status, item, detail="", hint=""):
    RESULTS.append(dict(status=status, item=item, detail=detail, hint=hint))
    line = f"[{status:7s}] {item}"
    if detail:
        line += f" -- {detail}"
    print(line)
    if hint and status != "PASS":
        print(f"          hint: {hint}")


def check_env():
    for mod in ("numpy", "pandas", "h5py"):
        try:
            m = __import__(mod)
            report("PASS", f"python package {mod}", getattr(m, "__version__", ""))
        except ImportError:
            report("FAIL", f"python package {mod}", "not installed", f"pip install {mod}")
    free = shutil.disk_usage(config.WS).free / 1e9 if os.path.exists(config.WS) else float("nan")
    report("PASS" if free > 5 else "WARN", "free disk on workspace", f"{free:.1f} GB",
           "ridge needs < 2 GB (aligned features + counts + preds)")
    report("PASS", "cpu count", str(os.cpu_count()))


def check_files():
    root = config.HER2ST_ROOT
    secs = glob.glob(os.path.join(root, "ST-cnts", "*.tsv*"))
    n = len({os.path.basename(s).split(".tsv")[0] for s in secs})
    report("PASS" if n == config.EXPECTED_N_SECTIONS else "FAIL", "her2st raw counts",
           f"{n} sections under {root}/ST-cnts", "set HER2ST_ROOT")
    lbl = glob.glob(os.path.join(root, "ST-pat", "lbl", "*_labeled_coordinates.tsv"))
    report("PASS" if len(lbl) >= 8 else "FAIL", "pathologist labels",
           f"{len(lbl)} files under {root}/ST-pat/lbl", "needed only for region_mean")
    if os.path.exists(config.PANEL):
        g = data.load_panel()
        report("PASS" if len(g) == config.EXPECTED_N_GENES else "FAIL", "gene panel",
               f"{len(g)} genes ({config.PANEL})")
    else:
        report("FAIL", "gene panel", f"{config.PANEL} missing", "set PANEL")
    gs = [n for n in ("all", "hvg", "svg", "marker")
          if os.path.exists(os.path.join(config.GENE_SETS, f"gene_set_{n}.txt"))]
    report("PASS" if len(gs) == 4 else "WARN", "gene sets", f"found {gs} in {config.GENE_SETS}",
           "scores per gene set are skipped for missing files")
    hints = {
        "index.json": "fixes the canonical spot order used by the cache-based ports; set BLEEP_INDEX_JSON",
        "expression.npy": "used only to verify the target definition; set HER2ST_EXPRESSION",
        "patches.npy": "not used by ridge (features come from existing caches); informational",
    }
    for p, what in [(config.INDEX_JSON, "shared cache index.json"),
                    (config.EXPRESSION_NPY, "shared cache expression.npy"),
                    (config.PATCHES_NPY, "shared cache patches.npy")]:
        if not os.path.exists(p):
            report("FAIL" if "index" in what else "WARN", what, f"{p} missing",
                   hints[what.split()[-1]])
            continue
        if p.endswith(".npy"):
            a = np.load(p, mmap_mode="r")
            report("PASS", what, f"shape {a.shape} {a.dtype}")
        else:
            with open(p) as f:
                j = json.load(f)
            ok = len(j.get("spot_keys", [])) == config.EXPECTED_N_SPOTS
            same_panel = list(j.get("panel", [])) == data.load_panel()
            report("PASS" if ok and same_panel else "FAIL", what,
                   f"{len(j.get('spot_keys', []))} spot keys, panel matches PANEL: {same_panel}")


COUNTS = {}          # read the 36 count files once per run


def check_index_and_truth():
    try:
        index = data.build_index()
    except Exception as e:  # noqa: BLE001
        report("FAIL", "canonical spot index", repr(e))
        return None
    ok = len(index) == config.EXPECTED_N_SPOTS and index.section.nunique() == config.EXPECTED_N_SECTIONS
    report("PASS" if ok else "FAIL", "canonical spot index",
           f"{len(index)} spots, {index.section.nunique()} sections, from {index.attrs['source']}")
    try:
        genes = data.load_panel()
        counts, lib_full = data.load_counts(index, genes, save=False)
        COUNTS["counts"], COUNTS["lib_full"] = counts, lib_full
        report("PASS", "raw counts aligned to index", f"{counts.shape}, zero-depth spots: "
               f"{int((counts.sum(1) == 0).sum())}")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "raw counts aligned to index", repr(e))
        return index
    if os.path.exists(config.EXPRESSION_NPY):
        E = np.load(config.EXPRESSION_NPY, mmap_mode="r")
        T = data.make_target("cpm_panel_ln", counts, lib_full, index.section.values)
        if E.shape == T.shape:
            d = float(np.abs(np.asarray(E, np.float64) - T).max())
            report("PASS" if d < 1e-4 else "FAIL", "shared truth == cpm_panel_ln(raw counts)",
                   f"max |diff| = {d:.2e}",
                   "if this fails, the cache rows are not in index.json order or the "
                   "target definition differs; do not run anything until resolved")
        else:
            report("FAIL", "shared truth shape", f"{E.shape} vs {T.shape}")
    labels = data.load_region_labels(index)
    lab = np.array([l is not None for l in labels])
    report("PASS" if lab.sum() > 3000 else "WARN", "region labels matched to spots",
           f"{int(lab.sum())} spots in {sorted(set(index.section.values[lab]))}",
           "expected 3,481 spots in A1 B1 C1 D1 E1 F1 G2 H1")
    return index


def check_sources(index):
    for name, meta in config.FEATURE_SOURCES.items():
        try:
            X, rep = features.align_source(name, index)
            ok = rep["dim"] == meta["dim"] and rep["nonfinite"] == 0
            report("PASS" if ok else "WARN", f"features: {name}",
                   f"{rep['files']} files, dim {rep['dim']}, all {rep['matched']} spots matched, "
                   f"{rep['extra']} extra, non-finite {rep['nonfinite']}  [{meta['desc']}]")
        except FileNotFoundError as e:
            hint = {"uni_native224": "build it with: python stflow/build_features.py "
                                     "--patch_mode native224  (GPU, a few minutes)"}.get(name, "")
            report("MISSING", f"features: {name}", str(e), hint)
        except Exception as e:  # noqa: BLE001
            report("FAIL", f"features: {name}", repr(e))
    report("PASS" if os.path.exists(config.UNI_CKPT) else "WARN", "UNI v1 weights",
           config.UNI_CKPT, "only needed to extract features that are missing")


def check_preds(pairs, index):
    if not pairs or index is None or "counts" not in COUNTS:
        return
    counts, lib_full = COUNTS["counts"], COUNTS["lib_full"]
    pos = dict(zip(index.key, index.row))
    for pair in pairs:
        name, run_dir = pair.split("=", 1)
        files = sorted(glob.glob(os.path.join(run_dir, "**", "preds", "*.npz"), recursive=True))
        if not files:
            report("MISSING", f"preds: {name}", f"no **/preds/*.npz under {run_dir}")
            continue
        z = np.load(files[0], allow_pickle=True)
        sec = os.path.basename(files[0])[:-4]
        if "spot_id" not in z.files:
            report("WARN", f"preds: {name}", f"{files[0]} has no spot_id; cannot verify target")
            continue
        rows = np.array([pos[data.make_key(sec, s)] for s in z["spot_id"]])
        sec_rows = np.where(index.section.values == sec)[0]
        # targets are computed on the WHOLE section (mediannorm needs the section median)
        full = {n: data.make_target(n, counts[sec_rows], lib_full[sec_rows], index.section.values[sec_rows])
                for n in data.TARGETS}
        where = {r: i for i, r in enumerate(sec_rows)}
        diffs = sorted(((n, float(np.abs(T[[where[r] for r in rows]] - z["truth"]).max()))
                        for n, T in full.items()), key=lambda t: t[1])
        best, d = diffs[0]
        report("PASS" if d < 1e-4 else "WARN", f"preds: {name} target space",
               f"{best} (max |diff| {d:.1e}) on {sec}; next: {diffs[1][0]} ({diffs[1][1]:.1e})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", nargs="*", default=[], metavar="MODEL=RUN_DIR")
    args = ap.parse_args(argv)
    print(f"== ridge workspace check (WORKSPACE={config.WS}) ==")
    check_env()
    check_files()
    index = check_index_and_truth()
    if index is not None:
        check_sources(index)
        check_preds(args.preds, index)
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    n_miss = sum(r["status"] == "MISSING" for r in RESULTS)
    print(f"== {n_fail} FAIL, {n_miss} MISSING, "
          f"{sum(r['status'] == 'WARN' for r in RESULTS)} WARN ==")
    out = os.path.join(config.RIDGE_CACHE, "workspace_check.json")
    try:
        os.makedirs(config.RIDGE_CACHE, exist_ok=True)
        with open(out, "w") as f:
            json.dump(RESULTS, f, indent=2)
        print(f"report -> {out}")
    except OSError as e:
        print(f"(could not write report: {e})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
