"""
tsize/manifest.py -- the single list of training subsets that BOTH pipelines read.

    python -m tsize.manifest --k7-dir ../results_4_models/ridge     # on the Mac (default)
    python -m tsize.manifest --k7-dir /workspace/ridge_results      # or on the pod

Writes tsize/manifest.json (commit it; the pod only reads it):

  ridge_subsets   for every test patient P (A-H) and k in {1,2,4,6,7}, EVERY size-k
                  subset of the 7 other patients: 7+21+35+7+1 = 71 per P, 568 in all.
                  subset_id = "<P>_<train patients>", e.g. "B_CDG".
  triplex_runs    for the 4 TRIPLEX test patients and k in {2,4}, 2 subsets drawn
                  (fixed seed) FROM the ridge list above, so every TRIPLEX run has a
                  ridge fit with identical train and test patients. k = 7 is the
                  existing triplex_lopo_833_e20 run (subset_id "<P>_<all 7>").
  ridge_configs   encoder x context x target, each with the alpha that the k = 7
                  inner leave-one-patient-out chose per test patient, read from
                  <k7-dir>/<tag>/fold_<P>/fold.json (the existing runs).

TRIPLEX test patients (pooled per-patient PCC of triplex_lopo_833_e20):
  B .364 (easiest), C .199 (second), A .077 (best of the hard three, so its gap is
  least compressed by a PCC floor), E .029 (E and F tie on PCC and gap; E is higher).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "manifest.json")
DEFAULT_K7_DIR = os.environ.get(
    "TSIZE_K7_DIR", os.path.join(HERE, "..", "..", "results_4_models", "ridge"))

PATIENTS = list("ABCDEFGH")
KS = [1, 2, 4, 6, 7]
SEED = 20261117

TRIPLEX_TEST = ["B", "C", "A", "E"]
TRIPLEX_KS = [2, 4]
TRIPLEX_PER_K = 2
TRIPLEX_K7_TAG = "triplex_lopo_833_e20"

# (source, context, target); each matches an existing k = 7 run whose alpha is reused.
RIDGE_CONFIGS = [
    ("uni_hest112", "local", "cpm_panel_ln"),
    ("uni_hest112", "local", "cp10k_full_log10"),
    ("uni_hest112", "local", "cp10k_panel_log10"),
    ("uni_hest112", "local", "mediannorm_panel_ln"),
    ("uni_hest112", "local", "raw_ln"),
    ("cigar", "local", "cpm_panel_ln"),          # TRIPLEX's frozen twin
    ("cigar", "full", "cpm_panel_ln"),           # same, with neighbourhood + global context
    ("resnet50", "local", "cp10k_full_log10"),
    ("ctranspath", "local", "mediannorm_panel_ln"),
]


def config_tag(source, context, target):
    """Same naming as ridge.run_lopo.default_tag, so the k = 7 run is found by name."""
    return f"ridge_{source}_{context}_{target}"


def subset_id(test, train):
    return f"{test}_{''.join(sorted(train))}"


def all_subsets(test, ks=KS):
    others = [p for p in PATIENTS if p != test]
    out = []
    for k in ks:
        for tr in itertools.combinations(others, k):
            out.append(dict(subset_id=subset_id(test, tr), test=test, k=k, train=list(tr)))
    return out


def draw_triplex(ridge_subsets, seed=SEED, tests=TRIPLEX_TEST, ks=TRIPLEX_KS, per_k=TRIPLEX_PER_K):
    """Draw `per_k` distinct subsets per (test, k) from the ridge list. Iteration order
    is fixed (tests, then ks, as listed), so the same seed always gives the same draw."""
    rng = np.random.default_rng(seed)
    runs = []
    for P in tests:
        for k in ks:
            pool = [s for s in ridge_subsets if s["test"] == P and s["k"] == k]
            pick = sorted(rng.choice(len(pool), per_k, replace=False).tolist())
            for i in pick:
                s = pool[i]
                runs.append(dict(s, tag=f"triplex_tsize_{s['subset_id']}"))
    return runs


def read_k7_alphas(k7_dir, tag):
    out = {}
    for P in PATIENTS:
        p = os.path.join(k7_dir, tag, f"fold_{P}", "fold.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no k = 7 alpha for {tag} fold {P}: {p}")
        with open(p) as f:
            fj = json.load(f)
        assert fj.get("patient") in (None, P), p
        out[P] = float(fj["alpha"])
    return out


def build(k7_dir=DEFAULT_K7_DIR, seed=SEED):
    ridge_subsets = [s for P in PATIENTS for s in all_subsets(P)]
    triplex_runs = draw_triplex(ridge_subsets, seed)
    configs = []
    for src, ctx, tgt in RIDGE_CONFIGS:
        tag = config_tag(src, ctx, tgt)
        configs.append(dict(tag=tag, source=src, context=ctx, target=tgt, k7_tag=tag,
                            alpha_k7=read_k7_alphas(k7_dir, tag)))
    return dict(
        seed=seed, patients=PATIENTS, ks=KS,
        alpha_modes={
            "k7": "alpha fixed to the k = 7 inner-LOPO choice for this test patient",
            "k7_per_spot": "k7 alpha * n_train_spots(subset) / n_train_spots(all 7 others); "
                           "same penalty per training spot (identical to k7 at k = 7)",
        },
        ridge_configs=configs,
        ridge_subsets=ridge_subsets,
        triplex=dict(test_patients=TRIPLEX_TEST, ks=TRIPLEX_KS, per_k=TRIPLEX_PER_K,
                     k7_tag=TRIPLEX_K7_TAG, runs=triplex_runs),
    )


def load(path=None):
    with open(path or os.environ.get("TSIZE_MANIFEST", DEFAULT_PATH)) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k7-dir", default=DEFAULT_K7_DIR,
                    help="folder holding the existing k = 7 ridge runs (<tag>/fold_<P>/fold.json)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=DEFAULT_PATH)
    a = ap.parse_args(argv)
    m = build(a.k7_dir, a.seed)
    with open(a.out, "w") as f:
        json.dump(m, f, indent=1)
    print(f"[manifest] {len(m['ridge_subsets'])} ridge subsets x {len(m['ridge_configs'])} configs, "
          f"{len(m['triplex']['runs'])} TRIPLEX runs -> {a.out}")
    for r in m["triplex"]["runs"]:
        print(f"  {r['tag']}  test={r['test']}  k={r['k']}  train={''.join(r['train'])}")


if __name__ == "__main__":
    main()
