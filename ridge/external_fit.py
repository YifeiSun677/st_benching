"""
ridge/external_fit.py -- the 8 her2st ridge folds as explicit weights, for the external cohorts.

The her2st runs (ridge.run_lopo) never store weights: each fold is refitted here on the SAME 7
training patients, SAME features, SAME target and the alpha that fold already chose (read from its
fold.json), so nothing is re-tuned and the external arms use exactly the model that was scored on
her2st.  verify() re-predicts every her2st held-out section and compares with the stored run.

Default = the reference baseline of Figs 2-5, on the scale the external arms are scored on:
    source  uni_hest112   UNI v1, 112 um crop resized to 224 px (= STFlow's He / TCGA features)
    context local
    target  cp10k_panel_ln     ln(1 + 1e4 * C / sum_panel C)  -- He truth is panel_cp10k_log1p
    alpha   from ridge_uni_hest112_local_cp10k_panel_log10/fold_<P>/fold.json
            (ln = ln(10) x log10, so the fitted pattern and the inner-CV PCC are identical; only the
            output scale differs: a ridge fit is linear in Y)

    python -m ridge.external_fit --verify        # local or pod; needs the aligned uni_hest112 cache
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import config, data, features, solver

ALPHA_TAG = "ridge_uni_hest112_local_cp10k_panel_log10"
SOURCE, TARGET = "uni_hest112", "cp10k_panel_ln"


def fit_folds(source=SOURCE, target=TARGET, alpha_tag=ALPHA_TAG, alpha_root=None):
    """-> dict patient -> dict(fold, alpha, W [d x G], b [G], trainmean [G]); genes = panel order."""
    alpha_root = alpha_root or config.OUTPUT_DIR
    index = data.build_index()
    genes = data.load_panel()
    counts, lib = data.load_counts(index, genes)
    Y = data.make_target(target, counts, lib, index.section.values, keys=index.key.values).astype(np.float64)
    X = features.load_aligned(source, index).astype(np.float64)
    pat = index.patient.values
    folds = {}
    for k, P in enumerate(config.PATIENTS):
        fj = os.path.join(alpha_root, alpha_tag, f"fold_{P}", "fold.json")
        alpha = float(json.load(open(fj))["alpha"])
        tr = pat != P
        fit = solver.RidgeFit(solver.Stats(X[tr], Y[tr]))
        W, b = fit.coef(alpha)
        folds[P] = dict(fold=k, alpha=alpha, W=W.astype(np.float32), b=b.astype(np.float32),
                        trainmean=fit.my.astype(np.float32), n_train=int(tr.sum()))
    meta = dict(source=source, target=target, context="local", alpha_tag=alpha_tag, genes=genes,
                dim=int(X.shape[1]))
    return folds, meta, (index, X, Y)


def predict(fold: dict, X) -> np.ndarray:
    return np.asarray(X, np.float64) @ fold["W"].astype(np.float64) + fold["b"].astype(np.float64)


def verify(folds, meta, cache, alpha_root=None):
    """Held-out her2st predictions vs the stored alpha_tag run (its target is log10: x ln 10)."""
    alpha_root = alpha_root or config.OUTPUT_DIR
    index, X, _ = cache
    worst = 0.0
    for P, f in folds.items():
        d = os.path.join(alpha_root, meta["alpha_tag"], f"fold_{P}", "preds")
        for sec in sorted(set(index.section[index.patient == P])):
            z = np.load(os.path.join(d, f"{sec}.npz"), allow_pickle=True)
            rows = index[(index.section == sec)]
            pos = {s: r for s, r in zip(rows.spot_id.astype(str), rows.row)}
            r = [pos[data.norm_spot_id(s)] for s in z["spot_id"]]
            mine = predict(f, X[r])
            ref = z["pred"].astype(np.float64) * np.log(10)
            worst = max(worst, float(np.abs(mine - ref).max()))
    ok = worst < 1e-3
    print(f"[verify] her2st held-out predictions vs {meta['alpha_tag']} (x ln 10): max |diff| {worst:.2e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    folds, meta, cache = fit_folds()
    for P, f in folds.items():
        print(f"fold {P}: alpha {f['alpha']:g}, n_train {f['n_train']}, W {f['W'].shape}")
    if a.verify and not verify(folds, meta, cache):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
