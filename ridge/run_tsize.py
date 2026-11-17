"""
ridge/run_tsize.py -- training-size experiment for ridge: for each held-out test
patient, fit on EVERY subset of k other patients (k in 1, 2, 4, 6, 7; the list is in
tsize/manifest.json) with alpha FIXED to the k = 7 inner-LOPO choice for that test
patient. Leaves ridge/run_lopo.py untouched.

    python -m ridge.run_tsize                               # every config in the manifest
    python -m ridge.run_tsize --configs ridge_cigar_local_cpm_panel_ln --tests B --ks 1 7
    python -m ridge.run_tsize --status                      # done / to-do counts, no fitting

Each fit is scored right away with the benchmark PCC and written as
    $RIDGE_OUT/tsize/<config tag>/<test>/<subset_id>.csv.gz
    columns: gene, alpha_mode, alpha, n_train, pcc, pcc_secavg
      pcc         per-gene PCC over ALL spots of the test patient, sections pooled (headline)
      pcc_secavg  per-gene PCC per section, nanmean over sections (= ridge/score.py)
      alpha_mode  k7           alpha of the k = 7 run for this test patient
                  k7_per_spot  k7 alpha * n_train / n_train(k = 7)  (same penalty per spot)
A fit whose file exists is skipped, so an interrupted run resumes where it stopped.
Files are written to a temp name and renamed, so a killed job never leaves half a file.

Same fit as run_lopo at k = 7: features standardised on the training patients only,
'global' PCA fitted on training sections only, closed-form ridge from summed
per-patient statistics, predictions cast to float32 before scoring (as stored preds).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from tsize import manifest as tsm          # run from the st_benching root, like ridge.*

from . import config, data, features, solver

ALPHA_MODES = ("k7", "k7_per_spot")


def out_root():
    return os.path.join(config.OUTPUT_DIR, "tsize")


def fit_path(tag, test, sid, root=None):
    return os.path.join(root or out_root(), tag, test, f"{sid}.csv.gz")


# ------------------------------------------------------------------ data per config
class Prepared:
    """Everything a config needs, loaded once: index, target Y, feature blocks."""

    def __init__(self, source, context, target, index=None, genes=None, counts=None, X=None):
        self.index = index if index is not None else data.build_index()
        self.genes = genes if genes is not None else data.load_panel()
        if counts is None:
            counts = data.load_counts(self.index, self.genes)
        C, lib_full = counts
        self.Y = data.make_target(target, C, lib_full, self.index.section.values,
                                  keys=self.index.key.values).astype(np.float64)
        X = X if X is not None else features.load_aligned(source, self.index)
        self.blocks = config.CONTEXTS[context]
        self.static = features.static_blocks(X, self.index, self.blocks)
        self.patient = self.index.patient.values
        self.sections = self.index.section.values
        self.has_global = "global" in self.blocks
        self._fixed = None if self.has_global else self.design(None)
        self._stats = {}

    def design(self, train_mask):
        """Feature matrix for one fit; identical to run_lopo.run_ridge."""
        parts = [self.static[b] for b in self.blocks if b != "global"]
        if self.has_global:
            parts.append(features.global_pca(self.static["global"], self.index,
                                             train_mask, config.GLOBAL_PCS))
        return np.hstack(parts).astype(np.float64)

    def stats(self, Xf, p, cache):
        if cache and p in self._stats:
            return self._stats[p]
        m = self.patient == p
        s = solver.Stats(Xf[m], self.Y[m])
        if cache:
            self._stats[p] = s
        return s


def check_split(test, train):
    """The rules every fit must satisfy (also used by the tests)."""
    train = list(train)
    if test in train:
        raise ValueError(f"test patient {test} is in the training set {train}")
    if not train or len(set(train)) != len(train):
        raise ValueError(f"bad training set {train}")
    bad = set(train + [test]) - set(config.PATIENTS)
    if bad:
        raise ValueError(f"unknown patients {bad}")
    return [p for p in config.PATIENTS if p in train]          # canonical order


def fit_subset(prep: Prepared, test, train, alphas):
    """Fit on `train`, predict `test`. Returns (test_rows, {alpha: pred float32}, n_train)."""
    train = check_split(test, train)
    train_mask = np.isin(prep.patient, train)
    test_rows = np.where(prep.patient == test)[0]
    assert not train_mask[test_rows].any()
    if prep.has_global:
        Xf = prep.design(train_mask)                      # PCA depends on the training sections
        st = [prep.stats(Xf, p, cache=False) for p in train]
    else:
        Xf = prep._fixed
        st = [prep.stats(Xf, p, cache=True) for p in train]
    fit = solver.RidgeFit(solver.Stats.total(st))
    B = fit.project(Xf[test_rows])
    preds = {a: fit.predict(None, a, B=B).astype(np.float32) for a in alphas}
    return test_rows, preds, int(train_mask.sum())


def score_pred(prep, test_rows, pred):
    p = pred.astype(np.float64)
    t = prep.Y[test_rows]
    return solver.gene_pcc(p, t), solver.sections_pcc(p, t, prep.sections[test_rows])


def alphas_for(cfg, prep, test, n_train):
    a7 = float(cfg["alpha_k7"][test])
    n7 = int((prep.patient != test).sum())
    return {"k7": a7, "k7_per_spot": a7 * n_train / n7}


def run_one(cfg, prep, sub, root=None):
    test, train = sub["test"], sub["train"]
    n_train = int(np.isin(prep.patient, train).sum())
    am = alphas_for(cfg, prep, test, n_train)
    rows, preds, n = fit_subset(prep, test, train, sorted(set(am.values())))
    assert n == n_train
    frames = []
    for mode in ALPHA_MODES:
        pcc, sec = score_pred(prep, rows, preds[am[mode]])
        frames.append(pd.DataFrame(dict(gene=prep.genes, alpha_mode=mode, alpha=am[mode],
                                        n_train=n_train, pcc=pcc, pcc_secavg=sec)))
    df = pd.concat(frames, ignore_index=True)
    path = fit_path(cfg["tag"], test, sub["subset_id"], root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, compression="gzip", float_format="%.9g")
    os.replace(tmp, path)
    return df


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None, help="default tsize/manifest.json")
    ap.add_argument("--configs", nargs="+", default=["all"], help="config tags, or 'all'")
    ap.add_argument("--tests", nargs="+", default=config.PATIENTS)
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--status", action="store_true", help="only count done / to-do fits")
    a = ap.parse_args(argv)

    m = tsm.load(a.manifest)
    cfgs = m["ridge_configs"] if a.configs == ["all"] else \
        [c for c in m["ridge_configs"] if c["tag"] in a.configs]
    unknown = set(a.configs) - {"all"} - {c["tag"] for c in m["ridge_configs"]}
    if unknown:
        ap.error(f"not in the manifest: {sorted(unknown)}")
    subs = [s for s in m["ridge_subsets"] if s["test"] in a.tests and (a.ks is None or s["k"] in a.ks)]

    todo_all = 0
    for cfg in cfgs:
        todo = [s for s in subs if not os.path.exists(fit_path(cfg["tag"], s["test"], s["subset_id"]))]
        print(f"[tsize] {cfg['tag']}: {len(subs) - len(todo)} done, {len(todo)} to do")
        todo_all += len(todo)
    if a.status or todo_all == 0:
        return

    index, genes = data.build_index(), data.load_panel()
    counts = data.load_counts(index, genes)
    X_cache = {}
    for cfg in cfgs:
        todo = [s for s in subs if not os.path.exists(fit_path(cfg["tag"], s["test"], s["subset_id"]))]
        if not todo:
            continue
        t0 = time.time()
        if cfg["source"] not in X_cache:
            X_cache = {cfg["source"]: features.load_aligned(cfg["source"], index)}   # keep one
        prep = Prepared(cfg["source"], cfg["context"], cfg["target"], index, genes, counts,
                        X_cache[cfg["source"]])
        cdir = os.path.join(out_root(), cfg["tag"])
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(cdir, "run_config.json"), "w") as f:
            json.dump(dict(cfg, seed=m["seed"], n_spots=len(index),
                           index_source=index.attrs["source"], global_pcs=config.GLOBAL_PCS,
                           alpha_modes=m["alpha_modes"]), f, indent=2)
        for i, s in enumerate(todo, 1):
            df = run_one(cfg, prep, s)
            if i == 1 or i % 25 == 0 or i == len(todo):
                r = df[df.alpha_mode == "k7"].pcc.mean()
                print(f"[tsize] {cfg['tag']} {i}/{len(todo)} {s['subset_id']} k={s['k']} "
                      f"pcc(k7 alpha)={r:.4f}  {time.time() - t0:.0f}s", flush=True)
    print(f"[tsize] done -> {out_root()}")


if __name__ == "__main__":
    main()
