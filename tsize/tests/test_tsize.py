"""
Tests for the training-size experiment. Run from the st_benching root:

    python -m pytest tsize/tests -q

Local (Mac or pod): manifest, split rules, a fully synthetic end-to-end check that
ridge.run_tsize at k = 7 with the stored alpha reproduces ridge.run_lopo exactly,
TRIPLEX split logic and the --dry-run. On the pod (aligned features in $RIDGE_CACHE
and the k = 7 runs in $RIDGE_OUT): the same reproduction on the REAL features of every
manifest config, against the stored preds and per_gene_pcc_by_fold.csv.
"""
from __future__ import annotations

import glob
import json
import os
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ridge import config, data, features, run_lopo, run_tsize, score, solver
from triplex import config as tconfig, her2st, run_tsize as trun
from tsize import aggregate, manifest as tsm

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("TEST_RESULTS", REPO.parent / "results_4_models"))
M = tsm.load()


# ------------------------------------------------------------------ manifest
def test_manifest_ridge_subsets():
    subs = M["ridge_subsets"]
    assert M["ks"] == [1, 2, 4, 6, 7] and M["patients"] == list("ABCDEFGH")
    assert len(subs) == 8 * 71 == 568
    assert len({s["subset_id"] for s in subs}) == 568
    for P in M["patients"]:
        mine = [s for s in subs if s["test"] == P]
        for k in M["ks"]:
            got = {tuple(s["train"]) for s in mine if s["k"] == k}
            others = [q for q in M["patients"] if q != P]
            assert got == set(combinations(others, k)), (P, k)
            assert len(got) == comb(7, k)
    for s in subs:
        assert s["test"] not in s["train"] and len(s["train"]) == s["k"]
        assert s["train"] == sorted(s["train"])
        assert s["subset_id"] == tsm.subset_id(s["test"], s["train"])


def test_manifest_triplex_draws_are_paired_and_reproducible():
    t = M["triplex"]
    assert t["test_patients"] == ["B", "C", "A", "E"] and t["ks"] == [2, 4]
    runs = t["runs"]
    assert len(runs) == 4 * 2 * 2 == 16
    ridge_ids = {s["subset_id"]: s for s in M["ridge_subsets"]}
    for r in runs:                                   # every TRIPLEX run has its ridge twin
        twin = ridge_ids[r["subset_id"]]
        assert (twin["test"], twin["k"], twin["train"]) == (r["test"], r["k"], r["train"])
        assert r["tag"] == f"triplex_tsize_{r['subset_id']}"
    for P in t["test_patients"]:
        for k in t["ks"]:
            ids = [r["subset_id"] for r in runs if r["test"] == P and r["k"] == k]
            assert len(ids) == len(set(ids)) == 2
    again = tsm.draw_triplex(M["ridge_subsets"], M["seed"])
    assert again == runs                             # same seed -> same draw
    assert tsm.draw_triplex(M["ridge_subsets"], M["seed"] + 1) != runs


def test_manifest_alphas():
    assert [(c["source"], c["context"], c["target"]) for c in M["ridge_configs"]] == \
        [tuple(c) for c in tsm.RIDGE_CONFIGS]
    for c in M["ridge_configs"]:
        assert set(c["alpha_k7"]) == set(M["patients"])
        assert all(a in config.ALPHAS for a in c["alpha_k7"].values()), c["tag"]
    if (RESULTS / "ridge").exists():                 # re-read from the k = 7 runs
        for c in M["ridge_configs"]:
            assert tsm.read_k7_alphas(RESULTS / "ridge", c["tag"]) == c["alpha_k7"]


# ------------------------------------------------------------------ held-out patient
def test_heldout_never_in_ridge_training():
    for s in M["ridge_subsets"]:
        tr = run_tsize.check_split(s["test"], s["train"])
        assert s["test"] not in tr
    for bad in (("B", ["A", "B"]), ("B", []), ("B", ["A", "A"]), ("B", ["Z"])):
        with pytest.raises(ValueError):
            run_tsize.check_split(*bad)


def _toy_spot_index():
    rows, i = [], 0
    for P in "ABCDEFGH":
        for k in (1, 2, 3):
            for _ in range(10 * k):
                rows.append(dict(section=f"{P}{k}", patient=P, cache_idx=i))
                i += 1
    return pd.DataFrame(rows)


def test_heldout_never_in_triplex_training():
    idx = _toy_spot_index()
    for r in M["triplex"]["runs"]:
        tr, te = her2st.split_sections(idx, r["test"], r["train"])
        assert {s[0] for s in tr} == set(r["train"]) and {s[0] for s in te} == {r["test"]}
        assert len(tr) == 3 * r["k"] and len(te) == 3
    with pytest.raises(ValueError):
        her2st.split_sections(idx, "B", ["A", "B"])
    with pytest.raises(ValueError):
        her2st.split_sections(idx, "B", ["Q"])


def test_triplex_default_split_unchanged():
    """train_patients=None must give exactly the old LOPO split."""
    idx = _toy_spot_index()
    for P in tconfig.PATIENTS:
        old_test = her2st.sections_for_patient(idx, P)
        old_train = [s for s in her2st.all_sections(idx) if s not in old_test]
        assert her2st.split_sections(idx, P) == (old_train, old_test)
        others = [q for q in tconfig.PATIENTS if q != P]
        assert her2st.split_sections(idx, P, others) == (old_train, old_test)


def test_triplex_dry_run(monkeypatch):
    ref = RESULTS / "triplex" / M["triplex"]["k7_tag"]
    if not (ref / "fold_A" / "run.json").exists():
        pytest.skip("no reference TRIPLEX run.json")
    idx = _toy_spot_index()
    monkeypatch.setattr(her2st, "load_spot_index", lambda *a, **k: idx)
    monkeypatch.setattr(tconfig, "OUTPUT_DIR", str(RESULTS / "triplex"))
    monkeypatch.setattr(tconfig, "BATCH_SIZE", 128)
    planned = trun.main(["--dry-run"])
    assert len(planned) == 16
    for p in planned:
        n = int(idx.patient.isin(p["train"]).sum())
        assert p["n_train_spots"] == n and p["steps_per_epoch"] == -(-n // 128)
        assert p["total_steps"] == p["steps_per_epoch"] * 20
    monkeypatch.setattr(tconfig, "LR", 3e-4)               # any change must be refused
    with pytest.raises(SystemExit):
        trun.main(["--dry-run"])


def test_triplex_epochs_must_match_reference():
    ref = RESULTS / "triplex" / M["triplex"]["k7_tag"]
    if not ref.exists():
        pytest.skip("no reference TRIPLEX run")
    with pytest.raises(SystemExit):
        trun.check_hparams(21, str(ref))


# ------------------------------------------------------------------ ridge reproduction
def _toy(seed=0, n_genes=15, d=10, grid=6):
    """Synthetic her2st-shaped data: 8 patients x 2 sections x grid^2 spots."""
    rng = np.random.default_rng(seed)
    keys = [f"{P}{s}:{x}x{y}" for P in "ABCDEFGH" for s in (1, 2)
            for x in range(grid) for y in range(grid)]
    index = data._index_from_keys(keys, "toy")
    n = len(index)
    X = rng.normal(size=(n, d)).astype(np.float32)
    lam = np.exp(X[:, :3] @ rng.normal(size=(3, n_genes)) * 0.5 + 1.0)
    counts = rng.poisson(lam).astype(np.float32)
    lib_full = counts.sum(1).astype(np.float64) + 5
    genes = [f"G{i}" for i in range(n_genes)]
    return index, genes, (counts, lib_full), X


@pytest.mark.parametrize("context", ["local", "full"])
def test_k7_fixed_alpha_reproduces_run_lopo_synthetic(tmp_path, monkeypatch, context):
    """run_lopo (inner-CV alpha) vs run_tsize at k = 7 with that alpha: same preds,
    same section-averaged PCC as ridge.score, pooled PCC as computed from the preds."""
    monkeypatch.setattr(config, "RIDGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(config, "GENE_SETS", str(tmp_path / "none"))
    monkeypatch.setitem(config.FEATURE_SOURCES, "toy", dict(desc="", twin_of=None,
                                                            target="cpm_panel_ln", dim=10))
    index, genes, counts, X = _toy()
    features.save_aligned("toy", X, index, {})
    Y = data.make_target("cpm_panel_ln", *counts, index.section.values).astype(np.float64)
    out = tmp_path / "k7"
    run_lopo.run_ridge(index, Y, genes, str(out), config.PATIENTS, {}, "toy", context, False)
    score.score_dir(str(out), quiet=True)
    stored = pd.read_csv(out / "per_gene_pcc_by_fold.csv", index_col=0)
    alphas = {P: json.loads((out / f"fold_{P}" / "fold.json").read_text())["alpha"]
              for P in config.PATIENTS}

    prep = run_tsize.Prepared("toy", context, "cpm_panel_ln", index, genes, counts, X)
    cfg = dict(tag=f"ridge_toy_{context}_cpm_panel_ln", alpha_k7=alphas)
    for P in config.PATIENTS:
        others = [q for q in config.PATIENTS if q != P]
        rows, preds, _ = run_tsize.fit_subset(prep, P, others, [alphas[P]])
        pred = preds[alphas[P]]
        ref = [np.load(f) for f in sorted(glob.glob(str(out / f"fold_{P}" / "preds" / "*.npz")))]
        ref_pred = np.vstack([z["pred"] for z in ref])
        assert np.abs(pred - ref_pred).max() < 1e-5
        df = run_tsize.run_one(cfg, prep, dict(test=P, train=others,
                                               subset_id=tsm.subset_id(P, others)),
                               root=str(tmp_path / "ts"))
        k7 = df[df.alpha_mode == "k7"].set_index("gene")
        assert (df.alpha_mode == "k7_per_spot").sum() == len(genes)
        assert np.allclose(df[df.alpha_mode == "k7_per_spot"].pcc, k7.pcc.values, equal_nan=True)
        assert np.allclose(k7.pcc_secavg.values, stored.loc[genes, P].values,
                           atol=1e-6, equal_nan=True)
        pooled = solver.gene_pcc(ref_pred.astype(np.float64),
                                 np.vstack([z["truth"] for z in ref]).astype(np.float64))
        assert np.allclose(k7.pcc.values, pooled, atol=1e-6, equal_nan=True)


def test_small_k_and_alpha_scaling_synthetic():
    index, genes, counts, X = _toy(1)
    prep = run_tsize.Prepared.__new__(run_tsize.Prepared)
    prep.__init__("toy", "local", "cpm_panel_ln", index, genes, counts, X)
    cfg = dict(tag="t", alpha_k7={P: 100.0 for P in config.PATIENTS})
    n7 = int((prep.patient != "B").sum())
    for train in (["A"], ["A", "C"], ["A", "C", "D", "E"]):
        n = int(np.isin(prep.patient, train).sum())
        a = run_tsize.alphas_for(cfg, prep, "B", n)
        assert a["k7"] == 100.0 and a["k7_per_spot"] == pytest.approx(100.0 * n / n7)
        rows, preds, n2 = run_tsize.fit_subset(prep, "B", train, [1.0])
        assert n2 == n and set(prep.patient[rows]) == {"B"}
        assert np.isfinite(preds[1.0]).all()
    with pytest.raises(ValueError):
        run_tsize.fit_subset(prep, "B", ["B", "C"], [1.0])


# ------------------------------------------------------------------ real data (pod)
K7_DIR = Path(os.environ.get("TSIZE_K7_DIR", config.OUTPUT_DIR))


def _real_ready(src):
    return (Path(config.RIDGE_CACHE) / f"features_{src}.npz").exists() and K7_DIR.exists()


@pytest.mark.parametrize("cfg", M["ridge_configs"], ids=[c["tag"] for c in M["ridge_configs"]])
def test_k7_fixed_alpha_reproduces_stored_run_real(cfg):
    """REAL features: k = 7 with the stored alpha == the existing run, all 8 folds.
    Preds to 1e-5, per-gene PCC (section-averaged vs the stored CSV, pooled vs the
    stored preds) to 1e-6."""
    if not _real_ready(cfg["source"]):
        pytest.skip(f"needs {config.RIDGE_CACHE}/features_{cfg['source']}.npz and {K7_DIR}")
    run_dir = K7_DIR / cfg["k7_tag"]
    if not (run_dir / "per_gene_pcc_by_fold.csv").exists():
        pytest.skip(f"no scored k = 7 run at {run_dir}")
    stored = pd.read_csv(run_dir / "per_gene_pcc_by_fold.csv", index_col=0)
    prep = run_tsize.Prepared(cfg["source"], cfg["context"], cfg["target"])
    for P in config.PATIENTS:
        others = [q for q in config.PATIENTS if q != P]
        a = cfg["alpha_k7"][P]
        rows, preds, _ = run_tsize.fit_subset(prep, P, others, [a])
        files = sorted(glob.glob(str(run_dir / f"fold_{P}" / "preds" / "*.npz")))
        ref = [np.load(f, allow_pickle=True) for f in files]
        keys = np.concatenate([[f"{z['section']}:{s}" for s in z["spot_id"]] for z in ref])
        order = {k: i for i, k in enumerate(prep.index.key.values[rows])}
        perm = np.array([order[k] for k in keys])
        pred = preds[a][perm]
        ref_pred = np.vstack([z["pred"] for z in ref])
        assert np.abs(pred - ref_pred).max() < 1e-5, P
        pcc, sec = run_tsize.score_pred(prep, rows, preds[a])
        assert np.allclose(sec, stored.loc[prep.genes, P].values, atol=1e-6, equal_nan=True), P
        pooled = solver.gene_pcc(ref_pred.astype(np.float64),
                                 np.vstack([z["truth"] for z in ref]).astype(np.float64))
        assert np.allclose(pcc, pooled, atol=1e-6, equal_nan=True), P


# ------------------------------------------------------------------ aggregation
def test_aggregate_on_fake_runs(tmp_path, monkeypatch):
    """Fake per-fit files + fake TRIPLEX preds -> tidy table, pairing and k = 1 matrix."""
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(6)]
    m = dict(M, ridge_configs=[c for c in M["ridge_configs"] if c["source"] == "cigar"][:1])
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(m))
    tag = m["ridge_configs"][0]["tag"]
    rdir, tdir = tmp_path / "ridge", tmp_path / "triplex"
    for s in m["ridge_subsets"]:
        p = rdir / tag / s["test"] / f"{s['subset_id']}.csv.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        base = 0.01 * s["k"]
        pd.DataFrame(dict(gene=genes * 2, alpha_mode=["k7"] * 6 + ["k7_per_spot"] * 6,
                          alpha=1.0, n_train=100, pcc=base, pcc_secavg=base)).to_csv(p, index=False)

    def fake_fold(d, gap):
        (d / "preds").mkdir(parents=True, exist_ok=True)
        t = rng.normal(size=(40, len(genes)))
        np.savez(d / "preds" / "X1.npz", pred=t + rng.normal(size=t.shape) * 0.1 + gap,
                 truth=t, genes=np.array(genes))
    for r in m["triplex"]["runs"]:
        fake_fold(tdir / r["tag"] / f"fold_{r['test']}", 0)
    for P in m["patients"]:
        fake_fold(tdir / m["triplex"]["k7_tag"] / f"fold_{P}", 0)

    res = aggregate.aggregate(str(rdir), str(tdir), str(tmp_path / "out"), str(mpath))
    tidy = res["tidy"]
    assert list(tidy.columns) == aggregate.TIDY_COLS
    assert len(tidy[tidy.model == tag]) == 568 * 2 * len(genes)
    assert len(tidy[tidy.model == "triplex"]) == (16 + 8) * len(genes)
    gap = res["gap"]
    assert len(gap[gap.alpha_mode == "k7"]) == 16 + 8
    one = gap[(gap.alpha_mode == "k7") & (gap.k == 2)].iloc[0]
    assert one.ridge == pytest.approx(0.02)                 # matched to the k = 2 ridge fit
    bk = res["by_k"]
    b = bk[(bk.model == tag) & (bk.alpha_mode == "k7") & (bk.test_patient == "B")]
    assert list(b.k) == [1, 2, 4, 6, 7] and list(b.n_subsets) == [7, 21, 35, 7, 1]
    assert b.delta_vs_k7.iloc[0] == pytest.approx(0.01 - 0.07)
    w = pd.read_csv(tmp_path / "out" / "k1_matrix" / f"{tag}__k7.csv", index_col=0)
    assert w.shape == (8, 8) and np.isnan(np.diag(w.values)).all()
    assert np.allclose(w.values[~np.eye(8, dtype=bool)], 0.01)
