"""
triplex/run_tsize.py -- the 16 TRIPLEX training-size runs listed in tsize/manifest.json
(4 test patients x k in {2, 4} x 2 subsets). Each run is one fold of train.train_fold
with an explicit training-patient list; every hyperparameter is the existing
triplex_lopo_833_e20 run's, which is checked before anything trains.

    python -m triplex.run_tsize --dry-run          # splits, spot counts, steps; no GPU, no torch
    python -m triplex.run_tsize                    # train + predict every run not yet done
    python -m triplex.run_tsize --tests B C        # a subset of the runs

Output (same layout as LOPO runs, so any scorer reads it):
    $TRIPLEX_OUT/triplex_tsize_<subset_id>/fold_<test>/preds/<SEC>.npz, run.json
Resume: a run whose final.pt AND preds exist is skipped; an interrupted run continues
from its resume.pt (train_fold(resume=True)), with the same RNG state.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from tsize import manifest as tsm          # run from the st_benching root, like triplex.*

from . import config, her2st

# run.json keys that must match the reference (k = 7) run exactly
MATCH_KEYS = ("epochs", "lr", "batch_size", "cpm", "smooth", "freeze_target",
              "pos_layer", "res_neighbor")


def current_hparams(epochs):
    return dict(epochs=epochs, lr=config.LR, batch_size=config.BATCH_SIZE, cpm=config.CPM,
                smooth=config.SMOOTH, freeze_target=config.FREEZE_TARGET,
                pos_layer=config.POS_LAYER, res_neighbor=list(config.RES_NEIGHBOR))


def reference_hparams(ref_dir):
    """Hyperparameters of the k = 7 run; every fold must agree with each other."""
    files = sorted(glob.glob(os.path.join(ref_dir, "fold_*", "run.json")))
    if not files:
        raise FileNotFoundError(f"no fold_*/run.json under {ref_dir}")
    ref = None
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        h = {k: r[k] for k in MATCH_KEYS}
        if ref is not None and h != ref:
            raise ValueError(f"{f}: folds of the reference run disagree: {h} vs {ref}")
        ref = h
    return ref


def check_hparams(epochs, ref_dir):
    ref = reference_hparams(ref_dir)
    cur = current_hparams(epochs)
    diff = {k: (cur[k], ref[k]) for k in MATCH_KEYS if cur[k] != ref[k]}
    if diff:
        raise SystemExit(f"[tsize] hyperparameters differ from {ref_dir} (now, ref): {diff}")
    return ref


def run_done(tag, patient):
    ck = os.path.join(config.CKPT_DIR, tag, f"fold_{patient}", "final.pt")
    preds = glob.glob(os.path.join(config.OUTPUT_DIR, tag, f"fold_{patient}", "preds", "*.npz"))
    return os.path.isfile(ck) and len(preds) > 0


def plan(runs, index_df, batch_size, epochs):
    """Split and step count of each run, without torch (for --dry-run and the tests)."""
    out = []
    for r in runs:
        tr, te = her2st.split_sections(index_df, r["test"], r["train"])
        n = int(index_df.section.isin(tr).sum())
        steps = -(-n // batch_size)                       # DataLoader, drop_last=False
        out.append(dict(r, train_sections=tr, test_sections=te, n_train_spots=n,
                        steps_per_epoch=steps, total_steps=steps * epochs))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--tests", nargs="+", default=None)
    ap.add_argument("--ks", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20,
                    help="must equal the reference run's (20); checked")
    ap.add_argument("--ref-dir", default=None,
                    help="k = 7 run to copy hyperparameters from "
                         "(default $TRIPLEX_OUT/<manifest k7_tag>)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    m = tsm.load(a.manifest)
    runs = [r for r in m["triplex"]["runs"]
            if (a.tests is None or r["test"] in a.tests) and (a.ks is None or r["k"] in a.ks)]
    ref_dir = a.ref_dir or os.path.join(config.OUTPUT_DIR, m["triplex"]["k7_tag"])
    ref = check_hparams(a.epochs, ref_dir)
    print(f"[tsize] hyperparameters match {ref_dir}: {ref}")

    index_df = her2st.load_spot_index()
    planned = plan(runs, index_df, config.BATCH_SIZE, a.epochs)
    n7 = {P: int((index_df.patient != P).sum()) for P in config.PATIENTS}
    for p in planned:
        state = "done" if run_done(p["tag"], p["test"]) else "todo"
        print(f"[tsize] {state}  {p['tag']:<24} test={p['test']} k={p['k']} "
              f"train={''.join(p['train'])}  {len(p['train_sections'])} sections, "
              f"{p['n_train_spots']} spots ({p['n_train_spots'] / n7[p['test']]:.0%} of k=7), "
              f"{p['total_steps']} steps")
    if a.dry_run:
        return planned

    from .train import train_fold                   # torch + upstream TRIPLEX only from here
    for p in planned:
        if run_done(p["tag"], p["test"]):
            continue
        print(f"\n[tsize] === {p['tag']} ===", flush=True)
        train_fold(p["test"], p["tag"], epochs=a.epochs, resume=True, train_patients=p["train"])
    print("[tsize] all runs done")
    return planned


if __name__ == "__main__":
    main()
