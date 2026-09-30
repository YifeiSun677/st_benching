#!/usr/bin/env python
"""TCGA stage B -- TRIPLEX (triplex_lopo_833_e20_ckpt, final.pt) on TCGA-BRCA whole-slide windows.

Same model, branches and APEG grid rule as run_triplex_he.py:
  target     224 px crop around the spot of the her2st-scale window JPEG, white pad (BLEEP's crop via
             her2st_dataset.Her2stSection.patch), ToTensor + ImageNet Normalize (run_triplex.run_model)
  global     CIGAR ResNet18 512-d feature of every spot's crop (not cached: 450k x 512 x 4 B = 0.9 GB)
  neighbour  triplex.build_features._build_neighbor on the WINDOW's array grid (5x5 offsets, centre =
             token 12): spots adjacent across a block boundary are adjacent in the tissue, so they stay
  position   tcga_common.her2st_blocks -- window coords translated to start at (2, 2), windows larger
             than her2st's range (x 2-32, y 2-34) split into blocks; each block is one forward pass, so
             the global branch sees a her2st-sized section.  --he-check feeds He's positions unchanged.
  APEG grid  recovered per fold exactly as run_triplex_he.py (--grid scan against the stored held-out
             her2st predictions; 'modal' or 'W,H' also accepted)
  output     raw = log1p(CPM over the panel) (BLEEP/TRIPLEX target space), columns reordered to the panel
             lin = max(expm1(raw), 0)
  no truth   the port's dataset reads ST-cnts only to list spots; placeholders keep every spot.

All 8 fold models stay on the GPU; sections are the outer loop.  Resumable per section.

--he-check SEC   one He section through THIS code path (fresh crops + CIGAR, He positions) vs the stored He
                 predictions (/workspace/runs/he_triplex/...): PASS = max |diff| < 1e-3.

writes: <out>/fold0<k>_<P>/agg/<SEC>.npz  (+ spots/<SEC>.npz for --spots-for patients)
usage:  cd /workspace/st_benching && python external/run_triplex_tcga.py [--limit 6] [--he-check BC23287_C1]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as K  # noqa: E402
import run_triplex as R  # noqa: E402  (BD with read_counts patch, grid helpers, run_model)
import tcga_common as T  # noqa: E402
from run_triplex import BD, TC, TH, TAG, TriTestSections, _load_section, build_cigar_encoder, encode_patches  # noqa: E402
from triplex.build_features import _build_neighbor  # noqa: E402

_find_image_her2st = BD.find_image


def _find_image_any(root, section):
    """TCGA and He sections live under ST-imgs/<patient id>/<SEC>/."""
    hits = sorted(glob.glob(os.path.join(root, "ST-imgs", "*", section, "*.jpg*")))
    return hits[0] if hits else _find_image_her2st(root, section)


BD.find_image = _find_image_any


def section_inputs(root, sec, panel_cache, enc, device):
    """Patches (uint8), CIGAR features, spot ids and spot table -- the port's own crop and encoder."""
    q = BD.Her2stCLIPDataset(str(root), [sec], panel_cache, is_train=False, verbose=False)
    S = q.sections[0]
    sid = [str(s) for s in S.spot_ids]
    patches = np.stack([S.patch(i) for i in range(len(sid))])
    glob_feat = encode_patches(enc, patches, device=device).astype(np.float32)
    return patches, glob_feat, sid


def run_block(model, patches, neigh, mask, glob_feat, pos, device):
    """R.run_model, safe for a 1-spot block.  Upstream squeezes size-1 dims, so a single spot reaches a
    LayerNorm as [512, 1] and crashes (He sections never had one spot; TCGA has isolated 1-spot windows).
    A 1-spot block is fed as two identical copies and the first row kept: attention over two identical
    tokens, APEG's per-cell averaging and per-token LayerNorm all give exactly the single-spot result."""
    if len(patches) == 1:
        d = lambda x: np.concatenate([x, x], 0)                              # noqa: E731
        return R.run_model(model, d(patches), d(neigh), d(mask), d(glob_feat), d(pos), device)[:1]
    return R.run_model(model, patches, neigh, mask, glob_feat, pos, device)


def force_eval(model):
    """Inference with dropout OFF (decision 2026-09-30).  Upstream MultiHeadAttention overrides train() and only
    calls super().train(mode) when attn_bias=True, so model.eval() leaves the bias-free attention layers (the
    global encoder: 3 layers x 6 submodules = 18) and their nn.Dropout in training mode -> every forward is a
    random draw (upstream's own inference, and our her2st / He / Visium TRIPLEX runs, had this).  model.eval()
    first (so the attn_bias layers build their cached self.ab), then clear the flag on whatever is still in
    training mode.  drop_p is zeroed too: it is only read by the flash branch, which the port disables, but
    this keeps inference deterministic if that ever changes.  Returns how many modules were forced."""
    model.eval()
    stuck = [m for m in model.modules() if m.training]
    for m in stuck:
        m.training = False
    for m in model.modules():
        if hasattr(m, "drop_p"):
            m.drop_p = 0.0
    assert not any(m.training for m in model.modules())
    return len(stuck)


def load_models(folds, panel_cache, grid_rule, device):
    models = {}
    for fd in folds:
        P, k = fd["patient"], fd["fold"]
        model, _ = R.load_fold(TAG, P, device=device)
        n_forced = force_eval(model)
        tr = {s: _load_section(s, panel_cache) for s in fd["train"]}
        modal, _ = R.pin_grid(model, [tr[s]["coords"] for s in fd["train"]])
        note = ""
        if grid_rule == "scan":
            te0 = TriTestSections(fd["test"][:1], panel_cache)
            st0 = np.load(os.path.join(TC.OUTPUT_DIR, TAG, f"fold_{P}", "preds", f"{fd['test'][0]}.npz"),
                          allow_pickle=True)["pred"]
            grid, rows = R.scan_grid(model, te0, fd["test"][0], st0, device)
            note = ", ".join(f"{g}:{d:.2g}" for g, d in rows[:3])
            if rows[0][1] > 1e-2:
                print(f"  [{P}] WARNING: no grid reproduces the stored predictions (best {rows[0][1]:.3g})", flush=True)
            del te0
        elif grid_rule == "modal":
            grid = modal
        else:
            grid = tuple(int(v) for v in grid_rule.split(","))
        R.set_grid(model, grid)
        models[P] = dict(fold=k, model=model,
                         extra=dict(ckpt=os.path.join(TC.CKPT_DIR, TAG, f"fold_{P}", "final.pt"), cohort="tcga",
                                    dropout="OFF at inference (force_eval): upstream train() override left "
                                            f"{n_forced} modules in training mode",
                                    apeg_grid=grid, apeg_grid_rule=grid_rule, apeg_grid_modal=modal, apeg_scan=note,
                                    neighbours="exact array-grid lookup on the whole window",
                                    positions="tcga_common.her2st_blocks (translate to 2,2; split to fit x 2-32, y 2-34)",
                                    raw="log1p(CPM over the panel)", inverse="max(expm1(raw), 0)", train=fd["train"]))
        print(f"  [{P}] dropout off ({n_forced} modules forced to eval); APEG grid {grid} "
              f"(rule {grid_rule}; modal {modal}) {note}", flush=True)
        del tr
        torch.cuda.empty_cache()
    return models


def he_check(sec, models, panel_cache, perm, enc, device, he_root):
    import he_common as H
    patches, glob_feat, sid = section_inputs(H.HE_DATA, sec, panel_cache, enc, device)
    sp = K.read_spots(sec, H.HE_DATA).loc[sid]
    neigh, mask = _build_neighbor(glob_feat, sp[["y", "x"]].to_numpy(np.int64))
    pos = sp[["x", "y"]].to_numpy(np.float32)
    ok = True
    for P, m in models.items():
        z = np.load(os.path.join(he_root, f"fold0{m['fold']}_{P}", "preds", f"{sec}.npz"), allow_pickle=True)
        assert [str(s) for s in z["spot_id"]] == sid, "spot order differs from the stored He predictions"
        pred = R.run_model(m["model"], patches, neigh, mask, glob_feat, pos, device)[:, perm]
        again = R.run_model(m["model"], patches, neigh, mask, glob_feat, pos, device)[:, perm]
        rep = float(np.abs(pred - again).max())
        c = float(np.corrcoef(pred.ravel(), z["pred"].ravel())[0, 1])
        ok = ok and rep == 0.0 and c > 0.99
        print(f"  he-check {sec} fold {P}: {len(sid)} spots, neighbours/spot {mask.sum(1).mean():.1f}, "
              f"run-twice max |diff| {rep:.2e}, vs stored (dropout-on) He: max |diff| "
              f"{float(np.abs(pred - z['pred']).max()):.2e}, corr {c:.6f}")
    print(f"HE-CHECK {'PASS' if ok else 'FAIL'} (dropout off: two runs bit-identical, and corr > 0.99 with the "
          "stored He predictions, which were random draws with dropout on)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--sections", nargs="*")
    ap.add_argument("--patients", nargs="*")
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--batch", nargs="*", type=int)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spots-for", nargs="*", default=None)
    ap.add_argument("--out", default="/workspace/runs/tcga_triplex")
    ap.add_argument("--grid", default="scan", help="'scan' (default, as He), 'modal', or 'W,H'")
    ap.add_argument("--he-check", default=None)
    ap.add_argument("--he-root", default="/workspace/runs/he_triplex")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out)
    he_root = os.path.abspath(a.he_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.chdir(str(K.ST_BENCH))                  # upstream init looks for ./weights/cigar (run.sh)
    panel = K.load_panel()
    panel_cache = TH.load_panel()
    assert set(panel_cache) == set(panel)
    perm = [panel_cache.index(g) for g in panel]

    folds = K.lopo_folds()
    if a.folds != "all":
        folds = [f for f in folds if f["patient"] in a.folds.split(",")]
    models = load_models(folds, panel_cache, a.grid, device)
    enc = build_cigar_encoder(device=device)
    print(f"loaded CIGAR encoder + {len(models)} fold models on {device}", flush=True)

    if a.he_check:
        he_check(a.he_check, models, panel_cache, perm, enc, device, he_root)
        return

    w = T.select_sections(a.sections, a.patients, a.kinds, a.batch, a.limit)
    spot_pats = set(T.default_spot_patients() if a.spots_for is None else a.spots_for)
    print(f"{len(w)} sections, {int(w.n_spots.sum())} spots; full spot matrices kept for {sorted(spot_pats)}")
    t_all, n_all = time.time(), 0
    for row in w.itertuples():
        dirs = {P: os.path.join(out_root, f"fold0{m['fold']}_{P}") for P, m in models.items()}
        if all(T.agg_path(d, row.section).exists() for d in dirs.values()):
            continue
        t0 = time.time()
        patches, glob_feat, sid = section_inputs(T.TCGA_DATA, row.section, panel_cache, enc, device)
        assert len(sid) == row.n_spots, f"{row.section}: dataset kept {len(sid)} of {row.n_spots} spots"
        sp = T.read_spots(row.section).loc[sid]
        neigh, mask = _build_neighbor(glob_feat, sp[["y", "x"]].to_numpy(np.int64))
        blocks = T.her2st_blocks(sp.x.values, sp.y.values)
        sds = []
        for P, m in models.items():
            raw = np.zeros((len(sid), len(panel)), np.float32)
            for idx, pos in blocks:
                raw[idx] = run_block(m["model"], patches[idx], neigh[idx], mask[idx], glob_feat[idx],
                                     pos.astype(np.float32), device)[:, perm]
            lin = np.maximum(np.expm1(raw.astype(np.float64)), 0.0)
            T.write_agg(dirs[P], row, pred_raw=raw, pred_lin=lin, spot_ids=sid, genes=panel, model="triplex",
                        fold=m["fold"], inverse="max(expm1(raw), 0)", save_spots=row.patient in spot_pats,
                        extra=m["extra"])
            sds.append(float(np.median(raw.std(0))) if len(sid) > 1 else float("nan"))
        torch.cuda.empty_cache()
        n_all += len(sid)
        print(f"{row.section}: {len(sid)} spots in {len(blocks)} block(s), neighbours/spot {mask.sum(1).mean():.1f}, "
              f"median across-spot SD of raw pred {np.nanmean(sds):.4f}, {time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t_all
    if n_all:
        print(f"DONE {n_all} spots x {len(models)} folds in {dt:.0f}s -> {1000 * dt / n_all:.2f} s per 1000 spots")


if __name__ == "__main__":
    main()
