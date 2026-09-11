"""
stflow_port/train.py -- train + predict ONE STFlow fold on her2st (833 panel, UNI features).

The optimisation loop, loss, flow-matching corruption, sampler and inference are the
released STFlow code (see stflow_import.py for the three minimal fixes). What differs
from upstream app/flow/train.py, and why, is in WORKING_PROCESS.md:
  * no test-set model selection: upstream evaluates the TEST split every epoch, keeps the
    best-test-PCC epoch and early-stops on it (patience 20). We train a fixed, pre-declared
    epoch budget and score the LAST epoch (benchmark-wide rule);
  * her2st LOPO folds instead of HEST-bench split CSVs;
  * per-spot predictions are saved (upstream only saves summary JSONs).

Modes
  LOPO fold      --test_patient A
  budget probe   --test_patient A --val_patient B --skip_test --epochs 200 --eval_every 5
                 (A is never touched; B is validation; trains on the other 6 patients)
  overfit check  --overfit_section B1 --epochs 300 --dropout 0 --attn_dropout 0
                 (train and predict the same section; wiring/alignment test)

Outputs  $RUNS_ROOT/<tag>/<fold_name>/
  run.json  loss_curve.csv  [probe_curve.csv budget.json]  last.pth
  preds/<SEC>.npz : pred, truth, train_mean, spot_id, coords, genes, section, patient,
                    epoch, eval_seed [, samples]
"""
import argparse
import contextlib
import csv
import io
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from dataset import EvalSet, TrainSet, load_cache, split_sections
from metrics import patient_metrics
from stflow_import import build_interpolant, git_commit, load


def set_random_seed(seed):            # = stflow.utils.set_random_seed (not imported: needs mygene)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run tag -> $RUNS_ROOT/<tag>/")
    ap.add_argument("--cache_tag", default="uni_v1_hest112")
    ap.add_argument("--test_patient", choices=C.PATIENTS)
    ap.add_argument("--val_patient", choices=C.PATIENTS)
    ap.add_argument("--overfit_section")
    ap.add_argument("--skip_test", action="store_true")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--n_samples", type=int, default=1,
                    help="extra prior draws saved for uncertainty analysis; pred = draw 0")
    ap.add_argument("--eval_seed", type=int, default=1234)
    ap.add_argument("--shuffle_test_features", action="store_true",
                    help="perturbation null: permute UNI features across spots of each test section")
    ap.add_argument("--prior_on_gpu", action="store_true",
                    help="sample the ZINB prior on the GPU (same distribution; upstream uses CPU)")
    ap.add_argument("--from_tag", default=None,
                    help="predict-only: load <RUNS_ROOT>/<from_tag>/<fold>/last.pth instead of training "
                         "(perturbation runs: grayscale cache or --shuffle_test_features)")
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
    for k, v in C.UPSTREAM.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    if not (a.test_patient or a.overfit_section):
        ap.error("need --test_patient or --overfit_section")
    return a


def fold_name(a):
    if a.overfit_section:
        return f"overfit_{a.overfit_section}"
    if a.val_patient:
        return f"probe_test{a.test_patient}_val{a.val_patient}"
    return f"fold{C.PATIENTS.index(a.test_patient):02d}_{a.test_patient}"


@torch.no_grad()
def predict(model, interp, sec_dicts, genes, a, seed, shuffle=False):
    """Upstream app/flow/test.py::test(), one loader per section; returns per-section preds."""
    U = load()
    loaders = [torch.utils.data.DataLoader(
        EvalSet(d, genes, shuffle_features_seed=(seed + i if shuffle else None)),
        batch_size=1, collate_fn=U["padding_batcher"]()) for i, d in enumerate(sec_dicts)]
    torch.manual_seed(seed)
    np.random.seed(seed)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        _, dump = U["upstream_test"](a, interp, model, loaders, return_all=True)
    out, o = [], 0
    for d in sec_dicts:
        n = len(d["labels"])
        p, t = dump["preds_all"][o:o + n], dump["targets_all"][o:o + n]
        if not np.array_equal(t, d["labels"]):
            raise RuntimeError(f"{d['section']}: row order changed inside upstream test()")
        out.append(p.astype(np.float32))
        o += n
    return out


def declare_budget(curve, ceiling):
    """Pre-declared rule (WORKING_PROCESS.md): smallest evaluated epoch whose validation
    mean per-gene PCC is within 0.005 of the curve maximum, rounded UP to a multiple of 10.
    If the curve is flat (max-min < 0.01), keep the upstream ceiling of 100."""
    ep = np.array([c["epoch"] for c in curve])
    pc = np.array([c["pcc_mean"] for c in curve])
    if pc.max() - pc.min() < 0.01:
        return dict(rule="flat curve -> upstream 100", epochs=100)
    first = int(ep[np.argmax(pc >= pc.max() - 0.005)])
    e = int(np.ceil(first / 10) * 10)
    return dict(rule="first epoch within 0.005 of max, ceil to 10", first_epoch=first,
                epochs=min(e, ceiling), max_pcc=float(pc.max()),
                max_epoch=int(ep[np.argmax(pc)]))


def main():
    a = parse()
    out = os.path.join(C.run_dir(a.tag), fold_name(a))
    rj = os.path.join(out, "run.json")
    if os.path.exists(rj) and not a.overwrite and json.load(open(rj)).get("status") == "complete":
        print(f"[train] {out} already complete - skipping (use --overwrite)")
        return
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    set_random_seed(a.seed)
    if not torch.cuda.is_available():
        print("[train] WARNING: no CUDA - running on CPU (tests only)")
        a.device = "cpu"
    device = a.device
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    data, genes, man = load_cache(a.cache_tag)
    a.n_genes = len(genes)
    a.feature_dim = C.UNI_FEATURE_DIM
    if a.overfit_section:
        train_s, val_s, test_s = [a.overfit_section], [], [a.overfit_section]
    else:
        train_s, val_s, test_s = split_sections(sorted(data), a.test_patient, a.val_patient)
        if not test_s:
            raise SystemExit(f"no cached sections for test patient {a.test_patient}")
        if a.val_patient and not val_s:
            raise SystemExit(f"no cached sections for val patient {a.val_patient}")
    print(f"[train] {fold_name(a)}  train={len(train_s)} secs  val={val_s}  test={test_s}  "
          f"genes={a.n_genes}  cache={a.cache_tag} ({man.get('patch_mode')})")

    U = load()
    model = U["Denoiser"](a).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    interp = build_interpolant(a)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    train_set = TrainSet([data[s] for s in train_s], a.patch_distribution, a.sample_times)
    loader = torch.utils.data.DataLoader(train_set, batch_size=a.batch_size,
                                         collate_fn=U["padding_batcher"]())
    train_mean = np.concatenate([data[s]["labels"] for s in train_s], 0).mean(0).astype(np.float32)
    print(f"[train] params={n_params:,}  steps/epoch={len(loader)}  epochs={a.epochs}")

    loss_rows, curve = [], []
    n_epochs_to_run = a.epochs
    if a.from_tag:
        src = os.path.join(C.run_dir(a.from_tag), fold_name(a))
        src_info = json.load(open(os.path.join(src, "run.json")))
        model.load_state_dict(torch.load(os.path.join(src, "last.pth"), map_location=device,
                                         weights_only=True), strict=True)
        a.epochs = int(src_info["args"]["epochs"])
        n_epochs_to_run = 0
        a.no_ckpt = True
        print(f"[train] predict-only from {src} (trained {a.epochs} epochs)")
    t_start = time.time()
    for epoch in range(1, n_epochs_to_run + 1):
        t0 = time.time()
        model.train()
        avg = 0.0
        for batch in loader:                                     # verbatim upstream loop
            batch = [x.to(device) for x in batch]
            img_features, coords, gene_exp = batch
            noisy_exp, t_steps = interp.corrupt_exp(gene_exp)
            _, loss = model(exp=noisy_exp, img_features=img_features, coords=coords,
                            labels=gene_exp, t_steps=t_steps)
            opt.zero_grad()
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_norm)
            opt.step()
            avg += loss.item()
        avg /= len(loader)
        dt = time.time() - t0
        loss_rows.append(dict(epoch=epoch, train_loss=avg, sec=round(dt, 2)))
        msg = f"  epoch {epoch:4d}  loss {avg:.4f}  {dt:.1f}s"
        if val_s and (epoch % a.eval_every == 0 or epoch == a.epochs):
            vp = predict(model, interp, [data[s] for s in val_s], genes, a, a.eval_seed)
            m, _, _ = patient_metrics(vp, [data[s]["labels"] for s in val_s], genes, C.MARKERS)
            curve.append(dict(epoch=epoch, train_loss=avg, **m))
            msg += (f"  | val {a.val_patient}: pcc {m['pcc_mean']:+.4f}  frac_pos {m['frac_pos']:.2f}"
                    f"  sse {m['sse_ratio_median']:.3f}  sd {m['sd_ratio_median']:.3f}"
                    f"  ERBB2 {m.get('pcc_ERBB2', float('nan')):+.3f}")
        print(msg, flush=True)
    t_train = time.time() - t_start

    if loss_rows:
        with open(os.path.join(out, "loss_curve.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(loss_rows[0]))
            w.writeheader(); w.writerows(loss_rows)
    budget = None
    if curve:
        with open(os.path.join(out, "probe_curve.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve[0]))
            w.writeheader(); w.writerows(curve)
        budget = declare_budget(curve, a.epochs)
        json.dump(budget, open(os.path.join(out, "budget.json"), "w"), indent=2)
        print(f"[train] BUDGET RULE -> {budget}")
    if not a.no_ckpt:
        torch.save(model.state_dict(), os.path.join(out, "last.pth"))

    t_pred, summary = 0.0, None
    if test_s and not a.skip_test:
        t1 = time.time()
        secs = [data[s] for s in test_s]
        draws = [predict(model, interp, secs, genes, a, a.eval_seed + k, a.shuffle_test_features)
                 for k in range(a.n_samples)]
        t_pred = time.time() - t1
        for i, d in enumerate(secs):
            extra = {"samples": np.stack([dr[i] for dr in draws]).astype(np.float16)} if a.n_samples > 1 else {}
            np.savez(os.path.join(out, "preds", f"{d['section']}.npz"),
                     pred=draws[0][i], truth=d["labels"], train_mean=train_mean,
                     spot_id=d["spot_id"], coords=d["coords"], genes=np.array(genes, dtype=str),
                     section=d["section"], patient=d["patient"], epoch=a.epochs,
                     eval_seed=a.eval_seed, **extra)
        summary, _, _ = patient_metrics([dr for dr in draws[0]], [d["labels"] for d in secs],
                                        genes, C.MARKERS)
        print(f"[train] TEST {test_s}: pcc_mean {summary['pcc_mean']:+.4f}  "
              f"median {summary['pcc_median']:+.4f}  frac_pos {summary['frac_pos']:.3f}  "
              f"sse {summary['sse_ratio_median']:.3f}  beat {summary['frac_genes_beat_baseline']:.3f}  "
              f"sd {summary['sd_ratio_median']:.3f}")

    peak = torch.cuda.max_memory_allocated(device) / 2**30 if torch.cuda.is_available() else 0.0
    info = dict(
        status="complete", fold=fold_name(a), args=vars(a),
        train_sections=train_s, val_sections=val_s, test_sections=test_s,
        n_params=n_params, steps_per_epoch=len(loader),
        sec_train=round(t_train, 1), sec_per_epoch=round(t_train / max(n_epochs_to_run, 1), 2),
        sec_predict=round(t_pred, 1), peak_gpu_gb=round(peak, 2),
        gpu=torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
        torch=torch.__version__, stflow_commit=git_commit(C.STFLOW_REPO),
        st_benching_commit=git_commit(C.ST_BENCH), cache_tag=a.cache_tag,
        patch_mode=man.get("patch_mode"), grayscale=man.get("grayscale"),
        test_summary=summary, budget=budget,
        final_train_loss=loss_rows[-1]["train_loss"] if loss_rows else None,
        from_tag=a.from_tag,
    )
    json.dump(info, open(rj, "w"), indent=2, default=str)
    print(f"[train] done {fold_name(a)}: {t_train/60:.1f} min train, {t_pred:.0f}s predict, "
          f"peak {peak:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
