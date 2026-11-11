"""
Local tests for the ridge baselines. Run from the st_benching root:

    python -m pytest ridge/tests -q

Tests that need the real her2st counts or saved model predictions are skipped
when those files are absent (e.g. on a fresh pod), so the suite runs anywhere.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ridge import config, data, features, run_lopo, score, solver

REPO = Path(__file__).resolve().parents[2]                 # st_benching/
SCIENCE = REPO.parent
HER2ST = Path(os.environ.get("TEST_HER2ST_ROOT", SCIENCE / "her2st" / "data"))
RESULTS = Path(os.environ.get("TEST_RESULTS", SCIENCE / "results_4_models"))
GENE_SETS = Path(os.environ.get("TEST_GENE_SETS", SCIENCE / "results" / "gene_sets"))

needs_her2st = pytest.mark.skipif(not (HER2ST / "ST-cnts").exists(), reason="no local her2st")
needs_results = pytest.mark.skipif(not RESULTS.exists(), reason="no local model results")


@pytest.fixture
def local_cfg(tmp_path, monkeypatch):
    """Point config at the local raw data and a throw-away output tree."""
    monkeypatch.setattr(config, "HER2ST_ROOT", str(HER2ST))
    monkeypatch.setattr(config, "PANEL", str(REPO / "panels" / "panel_833.txt"))
    monkeypatch.setattr(config, "GENE_SETS", str(GENE_SETS))
    monkeypatch.setattr(config, "INDEX_JSON", str(tmp_path / "no_index.json"))
    monkeypatch.setattr(config, "EXPRESSION_NPY", str(tmp_path / "no_expr.npy"))
    monkeypatch.setattr(config, "RIDGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    return tmp_path


# ------------------------------------------------------------------ pure units
def test_norm_spot_id():
    assert data.norm_spot_id("10x13") == "10x13"
    assert data.norm_spot_id("A1:10.0x13.0") == "10x13"
    assert data.norm_spot_id(b"9.98x20.02") == "10x20"


def test_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 20)) * rng.uniform(0.1, 5, 20) + 3
    Y = X[:, :3] @ rng.normal(size=(3, 4)) + rng.normal(size=(300, 4))
    for a in (0.1, 10.0, 1e4):
        fit = solver.RidgeFit(solver.Stats(X, Y))
        sk = Ridge(alpha=a).fit(StandardScaler().fit_transform(X), Y)
        Xs = (X - X.mean(0)) / X.std(0)
        assert np.allclose(fit.predict(X, a), sk.predict(Xs), atol=1e-8)
        W, b = fit.coef(a)
        assert np.allclose(X @ W + b, fit.predict(X, a), atol=1e-8)


def test_stats_are_additive():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(100, 6)), rng.normal(size=(100, 3))
    split = solver.Stats.total([solver.Stats(X[:40], Y[:40]), solver.Stats(X[40:], Y[40:])])
    whole = solver.Stats(X, Y)
    assert np.allclose(split.xx, whole.xx) and np.allclose(split.xy, whole.xy)
    assert np.allclose(solver.RidgeFit(split).predict(X, 3.0), solver.RidgeFit(whole).predict(X, 3.0))


def test_gene_pcc_nan_rule():
    p = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    t = np.array([[1.0, 1.0], [2.0, 0.0], [4.0, 2.0]])
    r = solver.gene_pcc(p, t)
    assert r[0] == pytest.approx(np.corrcoef(p[:, 0], t[:, 0])[0, 1])
    assert np.isnan(r[1])                                    # constant prediction -> NaN


def test_choose_alpha_behaviour():
    rng = np.random.default_rng(2)
    n, pats = 400, list("ABCD")
    patient = np.repeat(pats, n // 4)
    sections = np.array([f"{p}{1 + i % 2}" for i, p in enumerate(patient)])
    X = rng.normal(size=(n, 30))
    Y_signal = X[:, :2] @ rng.normal(size=(2, 5)) + 0.5 * rng.normal(size=(n, 5))
    a, scores, _, _ = solver.choose_alpha(X, Y_signal, patient, sections, pats, config.ALPHAS)
    assert np.nanmax(scores) > 0.5 and a < config.ALPHAS[-1]
    Y_noise = rng.normal(size=(n, 5))
    _, scores_n, _, _ = solver.choose_alpha(X, Y_noise, patient, sections, pats, config.ALPHAS)
    assert np.nanmax(scores_n) < 0.2                          # nothing to learn -> near 0


def test_align_source_detects_problems():
    index = data._index_from_keys(["S1:1x1", "S1:1x2", "S1:2x1"], "test")
    good = lambda: iter([("S1", ["1x1", "1x2", "2x1", "9x9"], np.eye(4, 3))])
    X, rep = features.align_source("t", index, reader=good)
    assert rep["extra"] == 1 and np.allclose(X, np.eye(3))
    missing = lambda: iter([("S1", ["1x1", "1x2"], np.eye(2, 3))])
    with pytest.raises(ValueError):
        features.align_source("t", index, reader=missing)
    dup = lambda: iter([("S1", ["1x1", "1.0x1.0", "1x2", "2x1"], np.eye(4, 3))])
    with pytest.raises(AssertionError):
        features.align_source("t", index, reader=dup)


def test_rings_and_context():
    keys = [f"S1:{x}x{y}" for x in range(5) for y in range(5)]
    index = data._index_from_keys(keys, "test")
    r1, r2 = data.ring_neighbours(index, 1), data.ring_neighbours(index, 2)
    centre = index.index[index.key == "S1:2x2"][0]
    corner = index.index[index.key == "S1:0x0"][0]
    assert len(r1[centre]) == 8 and len(r2[centre]) == 16
    assert len(r1[corner]) == 3
    X = np.arange(25, dtype=np.float32)[:, None]
    blk = features.ring_block(X, index, 1)
    assert blk[centre, 0] == pytest.approx(X[r1[centre], 0].mean()) and blk[centre, 1] == 8


# ------------------------------------------------------------------ real data
@needs_her2st
@needs_results
def test_targets_reproduce_each_port(local_cfg):
    """Every registered target reproduces the truth a real port stored (section A1)."""
    index = data.build_index()
    genes = data.load_panel()
    counts, lib_full = data.load_counts(index, genes, save=False)
    m = (index.section == "A1").values
    C, L, S = counts[m], lib_full[m], index.section.values[m]
    pos = {k.split(":")[1]: i for i, k in enumerate(index.key[m])}
    cases = {
        "cpm_panel_ln": "triplex/triplex_lopo_833_e20/fold_A/preds/A1.npz",
        "mediannorm_panel_ln": "path2space/path2space_lopo_833/preds/A1.npz",
        "raw_ln": "stflow/stflow_lopo_833_hest112_e90/fold00_A/preds/A1.npz",
    }
    for target, rel in cases.items():
        f = RESULTS / rel
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        rows = [pos[data.norm_spot_id(s)] for s in z["spot_id"]]
        T = data.make_target(target, C, L, S)[rows]
        assert np.abs(T - z["truth"]).max() < 1e-4, target
    # DeepPT stores targets without spot ids -> compare sorted row sums
    f = RESULTS / "deeppt" / "targets" / "A1.npy"
    if f.exists():
        best = data.identify_target(np.load(f), C, L, S, row_order_known=False)[0]
        assert best[0] == "cp10k_full_log10" and best[1] < 1e-3
    f = RESULTS / "histogene" / "htg_her2st_833_lopo_fold0.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        best = data.identify_target(z["truth"][z["section"] == "A1"], C, L, S, row_order_known=False)[0]
        assert best[0] == "cp10k_panel_log10" and best[1] < 1e-3


@needs_results
def test_scorer_reproduces_triplex_headline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GENE_SETS", str(GENE_SETS))
    run = RESULTS / "triplex" / "triplex_lopo_833_e20"
    ref = json.loads((run / "headline.json").read_text())
    h = score.score_dir(str(run), out_dir=str(tmp_path), quiet=True)
    assert h["pcc_mean"] == pytest.approx(ref["pcc_mean"], abs=1e-9)
    for k in ("hvg", "svg", "marker"):
        assert h[f"pcc_{k}"]["mean"] == pytest.approx(ref[f"pcc_{k}"]["mean"], abs=1e-9)
    mine = pd.read_csv(tmp_path / "per_gene_pcc_by_fold.csv", index_col=0)
    theirs = pd.read_csv(run / "per_gene_pcc_by_fold.csv", index_col=0)
    # triplex/score.py computes PCC in float32, this scorer in float64 -> ~1e-8 per gene
    assert np.allclose(mine[theirs.columns].values, theirs.values, equal_nan=True, atol=1e-6)


@needs_her2st
def test_end_to_end_on_real_counts(local_cfg, monkeypatch):
    """Full pipeline on the real 13,620 spots with SYNTHETIC features that carry a
    known amount of signal, for two folds."""
    index = data.build_index()
    genes = data.load_panel()
    counts, lib_full = data.load_counts(index, genes)
    Y = data.make_target("cpm_panel_ln", counts, lib_full, index.section.values)
    rng = np.random.default_rng(3)
    Yc = Y - Y.mean(0)
    _, _, vt = np.linalg.svd(Yc[rng.choice(len(Y), 3000, replace=False)], full_matrices=False)
    X = (Yc @ vt[:16].T) + 2.0 * rng.normal(size=(len(Y), 16))          # noisy expression PCs
    X = np.hstack([X, rng.normal(size=(len(Y), 48))]).astype(np.float32)  # + pure-noise dims
    monkeypatch.setitem(config.FEATURE_SOURCES, "synthetic",
                        dict(desc="test", twin_of=None, target="cpm_panel_ln", dim=64))
    features.save_aligned("synthetic", X, index, {"source": "synthetic"})

    pats = ["B", "C"]
    for argv in (["--model", "train_mean"], ["--model", "region_mean"],
                 ["--model", "ridge", "--source", "synthetic", "--context", "local"],
                 ["--model", "ridge", "--source", "synthetic", "--context", "full"]):
        monkeypatch.setattr(run_lopo.config, "FEATURE_SOURCES", config.FEATURE_SOURCES)
        tag = _run(argv + ["--patients"] + pats)
        h = score.score_dir(os.path.join(config.OUTPUT_DIR, tag), quiet=True)
        if argv[1] == "train_mean":
            assert h["pcc_mean"] is None and h["n_nan_genes"] == len(genes)
        elif argv[1] == "region_mean":
            secs = {os.path.basename(f)[:-4] for f in
                    glob.glob(os.path.join(config.OUTPUT_DIR, tag, "fold_*", "preds", "*.npz"))}
            assert secs == {"B1", "C1"}
            assert h["pcc_mean"] > 0.05
        else:
            assert h["pcc_mean"] > 0.1, (tag, h["pcc_mean"])
            fj = json.loads(Path(config.OUTPUT_DIR, tag, "fold_B", "fold.json").read_text())
            assert fj["alpha"] in config.ALPHAS and len(fj["inner_scores"]) == len(config.ALPHAS)
            assert set(fj["inner_by_patient"]) == set(config.PATIENTS) - {"B"}
    ann = score.score_dir(os.path.join(config.OUTPUT_DIR, "ridge_synthetic_local_cpm_panel_ln"),
                          sections=score.ANNOTATED, quiet=True)
    assert ann["sections"] == sorted(score.ANNOTATED)


def _run(argv):
    import sys
    old = sys.argv
    try:
        return run_lopo.main(argv)
    finally:
        sys.argv = old
