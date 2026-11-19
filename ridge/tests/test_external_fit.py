"""ridge.external_fit on synthetic data: the explicit-weight refit reproduces run_lopo's fold predictions,
and verify() catches a wrong alpha.   python -m pytest ridge/tests/test_external_fit.py -q"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from ridge import config, data, external_fit as EF, features, solver


@pytest.fixture
def fake(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for p in config.PATIENTS:
        for s in (1, 2):
            for i in range(30):
                rows.append((f"{p}{s}", f"{i}x{s}"))
    index = pd.DataFrame(rows, columns=["section", "spot_id"])
    index["patient"] = index.section.str[0]
    index["key"] = index.section + ":" + index.spot_id
    index["row"] = np.arange(len(index))
    genes = [f"G{j}" for j in range(12)]
    X = rng.normal(size=(len(index), 6)).astype(np.float32)
    C = rng.poisson(5 * np.exp(X @ rng.normal(size=(6, 12)) * .3)).astype(np.float32)
    monkeypatch.setattr(data, "build_index", lambda: index)
    monkeypatch.setattr(data, "load_panel", lambda: genes)
    monkeypatch.setattr(data, "load_counts", lambda idx, g: (C, C.sum(1).astype(np.float64)))
    monkeypatch.setattr(features, "load_aligned", lambda src, idx: X)
    # a stored her2st run on the log10 target, written the way run_lopo does (Stats.total over patients)
    Y10 = data.make_target("cp10k_panel_log10", C, C.sum(1), index.section.values).astype(np.float64)
    root = tmp_path / "ridge_results"
    for P in config.PATIENTS:
        tr = index.patient.values != P
        stats = [solver.Stats(X[index.patient.values == q], Y10[index.patient.values == q])
                 for q in config.PATIENTS if q != P]
        fit = solver.RidgeFit(solver.Stats.total(stats))
        fd = root / EF.ALPHA_TAG / f"fold_{P}"
        (fd / "preds").mkdir(parents=True)
        (fd / "fold.json").write_text(json.dumps(dict(alpha=30.0)))
        for sec in (f"{P}1", f"{P}2"):
            r = index.row.values[index.section.values == sec]
            np.savez(fd / "preds" / f"{sec}.npz", pred=fit.predict(X[r], 30.0).astype(np.float32),
                     spot_id=index.spot_id.values[r])
    return str(root)


def test_refit_reproduces_stored_run(fake):
    folds, meta, cache = EF.fit_folds(alpha_root=fake)
    assert set(folds) == set(config.PATIENTS) and meta["target"] == "cp10k_panel_ln"
    assert EF.verify(folds, meta, cache, alpha_root=fake)


def test_verify_catches_wrong_alpha(fake):
    folds, meta, cache = EF.fit_folds(alpha_root=fake)
    for P in folds:                                   # pretend the fold.json held another alpha
        json.dump(dict(alpha=1e4), open(os.path.join(fake, EF.ALPHA_TAG, f"fold_{P}", "fold.json"), "w"))
    bad, meta, cache = EF.fit_folds(alpha_root=fake)
    assert not EF.verify(bad, meta, cache, alpha_root=fake)
