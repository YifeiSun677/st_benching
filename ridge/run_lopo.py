"""
ridge/run_lopo.py -- leave-one-patient-out runs for the three linear baselines.

    python -m ridge.run_lopo --model train_mean
    python -m ridge.run_lopo --model region_mean
    python -m ridge.run_lopo --model ridge --source uni_hest112 --context local
    python -m ridge.run_lopo --model ridge --source cigar --context full --target native

--target    a name from data.TARGETS (default cpm_panel_ln = the shared expression.npy),
            or 'native' = the target space of the deep model this source is the twin of
--context   local | ring1 | rings | full   (see config.CONTEXTS)

Fairness rules (identical to the deep-model ports):
  * same 13,620 spots, same 833-gene panel, same 8 patient folds
  * the held-out patient is never used for fitting, scaling, PCA or choosing alpha
  * one hyperparameter (alpha) chosen by an inner leave-one-patient-out on the
    7 training patients; nothing is selected on the test patient
  * output contract = the other ports': <OUT>/<tag>/fold_<P>/preds/<SEC>.npz

Writes per fold: preds/<SEC>.npz and fold.json (alpha grid scores, chosen alpha).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import config, data, features, solver


# ------------------------------------------------------------------ shared
def _write_section(out_dir, sec, rows, pred, Y, index, genes, trainmean, meta):
    os.makedirs(os.path.join(out_dir, "preds"), exist_ok=True)
    sub = index.iloc[rows]
    pred = np.asarray(pred, np.float32)
    assert np.isfinite(pred).all(), f"{sec}: non-finite predictions"
    np.savez(os.path.join(out_dir, "preds", f"{sec}.npz"),
             pred=pred, truth=Y[rows].astype(np.float32), genes=np.array(genes),
             spot_id=sub.spot_id.to_numpy(dtype=str), coords=sub[["ax", "ay"]].values.astype(np.float32),
             section=sec, patient=sec[0], trainmean=np.asarray(trainmean, np.float32),
             **{k: np.array(v) for k, v in meta.items()})


def _sections_of(index, rows):
    return list(dict.fromkeys(index.section.values[rows]))


# ------------------------------------------------------------------ baselines
def run_train_mean(index, Y, genes, out_root, patients, meta):
    for P in patients:
        test = np.where(index.patient.values == P)[0]
        mu = Y[index.patient.values != P].mean(0)
        fdir = os.path.join(out_root, f"fold_{P}")
        for sec in _sections_of(index, test):
            rows = test[index.section.values[test] == sec]
            _write_section(fdir, sec, rows, np.tile(mu, (len(rows), 1)), Y, index, genes, mu, meta)
        _dump(fdir, dict(patient=P, model="train_mean"))
        print(f"[train_mean] fold {P}: {len(test)} spots")


def run_region_mean(index, Y, genes, out_root, patients, meta):
    labels = data.load_region_labels(index)
    labelled = np.array([l is not None for l in labels])
    print(f"[region_mean] labelled spots: {labelled.sum()} in "
          f"{sorted(set(index.section.values[labelled]))}")
    for P in patients:
        is_test = index.patient.values == P
        tr = labelled & ~is_test
        mu_all = Y[~is_test].mean(0)
        means = {lab: Y[tr & (labels == lab)].mean(0) for lab in set(labels[tr])}
        test = np.where(labelled & is_test)[0]
        fallback = sum(labels[r] not in means for r in test)
        pred = np.vstack([means.get(labels[r], mu_all) for r in test])
        fdir = os.path.join(out_root, f"fold_{P}")
        for sec in _sections_of(index, test):
            m = index.section.values[test] == sec
            _write_section(fdir, sec, test[m], pred[m], Y, index, genes, mu_all,
                           dict(meta, labels=labels[test[m]].astype(str)))
        _dump(fdir, dict(patient=P, model="region_mean", n_test=int(len(test)),
                         train_label_counts={k: int((tr & (labels == k)).sum()) for k in means},
                         fallback_to_train_mean=int(fallback)))
        print(f"[region_mean] fold {P}: {len(test)} labelled test spots, "
              f"{len(means)} labels in training, {fallback} fallbacks")


def run_ridge(index, Y, genes, out_root, patients, meta, source, context, clip):
    X = features.load_aligned(source, index)
    blocks = config.CONTEXTS[context]
    static = features.static_blocks(X, index, blocks)
    patient = index.patient.values
    sections = index.section.values
    alphas = list(config.ALPHAS)
    for P in patients:
        t0 = time.time()
        train_mask = patient != P
        parts = [static[b] for b in blocks if b != "global"]
        if "global" in blocks:
            parts.append(features.global_pca(static["global"], index, train_mask, config.GLOBAL_PCS))
        Xf = np.hstack(parts).astype(np.float64)
        train_patients = [p for p in config.PATIENTS if p != P]

        best, scores, per_inner, stats = solver.choose_alpha(
            Xf, Y, patient, sections, train_patients, alphas)
        fit = solver.RidgeFit(solver.Stats.total([stats[p] for p in train_patients]))
        test = np.where(~train_mask)[0]
        pred = fit.predict(Xf[test], best)
        if clip:
            pred = np.clip(pred, 0, None)

        fdir = os.path.join(out_root, f"fold_{P}")
        for sec in _sections_of(index, test):
            m = sections[test] == sec
            _write_section(fdir, sec, test[m], pred[m], Y, index, genes, fit.my,
                           dict(meta, alpha=best))
        edge = solver.edge_status(alphas, scores, best)
        _dump(fdir, dict(patient=P, model="ridge", source=source, context=context,
                         blocks=blocks, n_features=int(Xf.shape[1]), n_train=int(train_mask.sum()),
                         alpha=best, alpha_at_grid_edge=edge, alphas=alphas,
                         inner_scores=[float(s) for s in scores],
                         inner_by_patient={q: [float(v) for v in s] for q, s in per_inner.items()}))
        print(f"[ridge {source}/{context}] fold {P}: d={Xf.shape[1]} alpha={best:g} "
              f"inner={np.nanmax(scores):.4f} {edge} ({time.time() - t0:.0f}s)")


def _dump(fdir, obj):
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, "fold.json"), "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------------------------------------------ CLI
def default_tag(model, source, context, target):
    if model == "ridge":
        return f"ridge_{source}_{context}_{target}"
    return f"{model}_{target}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["train_mean", "region_mean", "ridge"])
    ap.add_argument("--source", choices=sorted(config.FEATURE_SOURCES))
    ap.add_argument("--context", default="local", choices=sorted(config.CONTEXTS))
    ap.add_argument("--target", default="cpm_panel_ln")
    ap.add_argument("--patients", nargs="+", default=config.PATIENTS)
    ap.add_argument("--clip", action="store_true", default=config.CLIP_AT_ZERO,
                    help="clip predictions at 0 (off by default: pure linear output)")
    ap.add_argument("--tag")
    args = ap.parse_args(argv)

    if args.model == "ridge" and not args.source:
        ap.error("--model ridge needs --source")
    target = args.target
    if target == "native":
        if not args.source:
            ap.error("--target native needs --source")
        target = config.FEATURE_SOURCES[args.source]["target"]
    tag = args.tag or default_tag(args.model, args.source, args.context, target)
    out_root = os.path.join(config.OUTPUT_DIR, tag)

    index = data.build_index()
    genes = data.load_panel()
    counts, lib_full = data.load_counts(index, genes)
    Y = data.make_target(target, counts, lib_full, index.section.values).astype(np.float64)
    print(f"[run] {tag}: {len(index)} spots ({index.attrs['source']}), {len(genes)} genes, "
          f"target={target}")

    meta = dict(model=args.model, target=target, source=args.source or "",
                context=args.context if args.model == "ridge" else "")
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "run_config.json"), "w") as f:
        json.dump(dict(vars(args), target=target, tag=tag, n_spots=len(index),
                       index_source=index.attrs["source"], alphas=config.ALPHAS,
                       global_pcs=config.GLOBAL_PCS), f, indent=2)

    if args.model == "train_mean":
        run_train_mean(index, Y, genes, out_root, args.patients, meta)
    elif args.model == "region_mean":
        run_region_mean(index, Y, genes, out_root, args.patients, meta)
    else:
        run_ridge(index, Y, genes, out_root, args.patients, meta,
                  args.source, args.context, args.clip)
    print(f"[run] done -> {out_root}\n      score with: python -m ridge.score --tag {tag}")
    return tag


if __name__ == "__main__":
    main()
