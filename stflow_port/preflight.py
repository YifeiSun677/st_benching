"""
stflow_port/preflight.py -- every check that must pass before spending GPU time.
Prints PASS / WARN / FAIL per check and exits non-zero on any FAIL.

  python stflow_port/preflight.py                 # all stages (cache stage auto-skips if absent)
  python stflow_port/preflight.py --stage env model
  python stflow_port/preflight.py --stage cache --cache_tag uni_v1_hest112

Stages
  env    python/torch/CUDA/GPU, required packages present, forbidden ones absent
  model  clone present + pinned commit; FIX 1-3 verified; 833-gene forward/backward;
         ZINB prior moments vs analytic values
  data   her2st layout, 36 sections / 8 patients, spot joins, integer counts,
         panel (833 unique, coverage), um/px per section
  uni    UNI checkpoint loads strictly, outputs 1024-d
  cache  per-section arrays consistent; labels + spot order match a fresh read of
         ST-cnts; UNI features for 8 random spots recomputed and matched (row alignment);
         EvalSet preserves row order; one real training step on GPU
"""
import argparse
import importlib
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

RESULTS = []


def report(status, name, detail=""):
    RESULTS.append(status)
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def check(name):
    def deco(fn):
        def run(*a, **k):
            try:
                fn(*a, **k)
            except Exception as e:
                report("FAIL", name, f"{type(e).__name__}: {e}")
                traceback.print_exc(limit=2)
        return run
    return deco


# --------------------------------------------------------------------- env
@check("env")
def stage_env():
    import torch
    report("PASS", "python", sys.version.split()[0])
    report("PASS", "torch", f"{torch.__version__}  cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        report("PASS", "gpu", f"{p.name}  {p.total_memory/2**30:.1f} GB")
    else:
        report("FAIL", "gpu", "CUDA not available")
    for mod in ["timm", "einops", "torch_geometric", "scanpy", "h5py", "scipy", "pandas",
                "PIL", "huggingface_hub"]:
        try:
            m = importlib.import_module(mod)
            report("PASS", f"import {mod}", getattr(m, "__version__", ""))
        except Exception as e:
            report("FAIL", f"import {mod}", str(e))
    import timm
    from timm.models import vision_transformer as vt
    report("PASS" if hasattr(vt, "SwiGLUPacked") else "FAIL", "timm has SwiGLUPacked",
           f"timm {timm.__version__} (need >=0.9)")
    for bad in ["scprep", "scvi"]:
        try:
            importlib.import_module(bad)
            report("WARN", f"{bad} installed", "not needed; scprep pins pandas<2.1 - uninstall if it breaks pandas")
        except Exception:
            report("PASS", f"{bad} not installed", "not needed by the port")
    for p in [C.WS, C.RUNS_ROOT, C.CACHE_ROOT]:
        os.makedirs(p, exist_ok=True)
    st = os.statvfs(C.WS)
    free = st.f_bavail * st.f_frsize / 2**30
    report("PASS" if free > 10 else "WARN", "free space on volume", f"{free:.1f} GB at {C.WS}")


# --------------------------------------------------------------------- model
@check("model")
def stage_model():
    import argparse
    import torch
    from stflow_import import load, make_zinb_sampler, git_commit
    U = load()
    commit = git_commit(C.STFLOW_REPO)
    report("PASS" if commit == C.STFLOW_PINNED_COMMIT else "WARN", "STFlow clone commit",
           f"{commit[:10]} (pinned {C.STFLOW_PINNED_COMMIT[:10]})")

    # FIX 3: identical ops
    fa = U["FA"].FrameAveraging(dim=2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig = U["FA"].FrameAveraging._original_create_ops(fa, 2)
    report("PASS" if torch.equal(orig, fa.ops) else "FAIL", "FIX 3 frame ops identical",
           str(fa.ops.tolist()))

    def mk(G):
        ns = argparse.Namespace(**C.UPSTREAM)
        ns.n_genes = G
        return U["Denoiser"](ns)

    m50 = mk(50)
    in50 = m50.backbone.blks[0].attn._stflow_port_attn_in
    exp50 = (C.UPSTREAM["hidden_dim"] // C.UPSTREAM["n_heads"]) * 2 + \
            C.UPSTREAM["pairwise_hidden_dim"] // C.UPSTREAM["n_heads"] + 50
    report("PASS" if in50 == exp50 == 146 else "FAIL", "FIX 2 is identity at 50 genes",
           f"attn-MLP in_features={in50} (released hard-codes 146)")
    report("PASS", "FIX 1 GeneUpdate builds", "Denoiser constructed without TypeError")

    G = C.EXPECTED_N_GENES
    m = mk(G)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev)
    n_params = sum(p.numel() for p in m.parameters())
    N = 600
    x = torch.randn(2, N, G, device=dev)
    f = torch.randn(2, N, 1024, device=dev)
    f[1, 400:] = 0                                   # padding rows, as padding_batcher does
    c = torch.rand(2, N, 2, device=dev) * 5000
    y = torch.rand(2, N, G, device=dev)
    pred, loss = m(x, f, c, y, torch.rand(2, device=dev))
    loss.backward()
    ok = pred.shape == (2, N, G) and torch.isfinite(loss).item() and \
        all(p.grad is not None for p in m.parameters() if p.requires_grad)
    report("PASS" if ok else "FAIL", "833-gene forward/backward (FIX 2)",
           f"pred {tuple(pred.shape)}  loss {loss.item():.3f}  params {n_params:,}")

    s = make_zinb_sampler(C.UPSTREAM["zinb_total_count"], C.UPSTREAM["zinb_logits"],
                          C.UPSTREAM["zinb_zi_logits"])
    torch.manual_seed(0)
    z = s((1000, 1000)).double()
    an = s.analytic
    em, ev, ez = z.mean().item(), z.var().item(), (z == 0).double().mean().item()
    ok = abs(em - an["mean"]) < 0.01 and abs(ev - an["var"]) < 0.03 and abs(ez - an["p_zero"]) < 0.005
    report("PASS" if ok else "FAIL", "ZINB prior moments",
           f"mean {em:.4f}/{an['mean']:.4f}  var {ev:.4f}/{an['var']:.4f}  "
           f"P0 {ez:.4f}/{an['p_zero']:.4f}  shape {tuple(z.shape)}")


# --------------------------------------------------------------------- data
@check("data")
def stage_data():
    import her2st_io as H
    for sub in ["ST-cnts", "ST-imgs", "ST-spotfiles"]:
        p = os.path.join(C.HER2ST_ROOT, sub)
        report("PASS" if os.path.isdir(p) else "FAIL", f"her2st {sub}/", p)
    secs = H.list_sections()
    pats = sorted({s[0] for s in secs})
    report("PASS" if len(secs) == C.EXPECTED_N_SECTIONS and pats == C.PATIENTS else "FAIL",
           "sections", f"{len(secs)} sections, patients {''.join(pats)}")
    genes = H.load_panel()
    report("PASS" if len(genes) == C.EXPECTED_N_GENES and len(set(genes)) == len(genes) else "FAIL",
           "panel", f"{len(genes)} genes ({len(set(genes))} unique) first={genes[:4]} from {C.PANEL}")
    seen, total, umpps, rows = set(), 0, [], []
    for s in secs:
        cnt, spt = H.load_section(s)
        rawc = H.read_counts(s)
        seen |= set(cnt.columns)
        vals = cnt.to_numpy()
        is_int = np.allclose(vals[:50], np.round(vals[:50]))
        umpp, sx, sy = H.estimate_um_per_px(spt)
        img = H.image_path(s)
        dropped = len(rawc) - len(cnt)
        total += len(cnt)
        umpps.append(umpp)
        rows.append((s, len(cnt), dropped, umpp, is_int, os.path.basename(img)))
        if not is_int:
            report("FAIL", f"{s} raw counts", "non-integer values: ST-cnts should hold RAW counts")
        if dropped:
            report("WARN", f"{s} spot join", f"{dropped} ST-cnts spots have no spot-file row (dropped)")
    print("    section  n_spots  um/px   crop_px(112um)")
    for s, n, d, u, i, img in rows:
        print(f"    {s:7s} {n:7d}  {u:6.3f}  {int(round(C.HEST_PATCH_UM / u)):5d}")
    report("PASS" if total == C.EXPECTED_N_SPOTS else "WARN", "total spots",
           f"{total} (benchmark uses {C.EXPECTED_N_SPOTS})")
    cov = sum(g in seen for g in genes)
    report("PASS" if cov >= 0.9 * len(genes) else "FAIL", "panel coverage in ST-cnts",
           f"{cov}/{len(genes)} panel genes present in >=1 section "
           f"(if ~0: panel is ENSG but ST-cnts are symbols, or vice versa)")
    u = np.array(umpps)
    spread = (u.max() - u.min()) / np.median(u)
    report("PASS" if 0.2 < np.median(u) < 2.0 else "WARN", "um/px plausible",
           f"median {np.median(u):.3f}  range {u.min():.3f}-{u.max():.3f}  (assumes 200 um pitch)")
    report("PASS" if spread < 0.15 else "WARN", "um/px consistent across sections",
           f"relative spread {spread:.2%}")


# --------------------------------------------------------------------- uni
@check("uni")
def stage_uni():
    import torch
    from build_features import load_uni
    size = os.path.getsize(C.UNI_CKPT) / 2**30
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_uni(dev)
    with torch.inference_mode():
        o = m(torch.zeros(2, 3, 224, 224, device=dev))
    report("PASS" if o.shape == (2, 1024) else "FAIL", "UNI strict load + forward",
           f"{C.UNI_CKPT} ({size:.2f} GB) -> {tuple(o.shape)}")


# --------------------------------------------------------------------- cache
@check("cache")
def stage_cache(tag):
    import torch
    from PIL import Image
    import her2st_io as H
    from build_features import crop, to_uni_input, load_uni
    from dataset import load_cache, EvalSet, TrainSet
    d = C.cache_dir(tag)
    if not os.path.exists(os.path.join(d, "manifest.json")):
        report("WARN", "cache", f"{d} not built yet - run build_features.py, then rerun --stage cache")
        return
    data, genes, man = load_cache(tag)
    n = sum(len(v["labels"]) for v in data.values())
    report("PASS" if len(data) == C.EXPECTED_N_SECTIONS else "FAIL", "cache sections",
           f"{len(data)} sections, {n} spots, mode={man.get('patch_mode')} gray={man.get('grayscale')}")
    bad = [s for s, v in data.items() if not (np.isfinite(v["features"]).all()
           and v["features"].shape[1] == 1024 and v["labels"].shape[1] == len(genes))]
    report("PASS" if not bad else "FAIL", "cache shapes/finite", f"bad={bad}")

    rng = np.random.default_rng(0)
    sec = sorted(data)[int(rng.integers(len(data)))]
    cnt, spt = H.load_section(sec)
    lab, _, _ = H.compute_targets(cnt, genes)
    ok = list(cnt.index) == list(data[sec]["spot_id"]) and np.array_equal(lab, data[sec]["labels"])
    report("PASS" if ok else "FAIL", f"labels+spot order vs fresh ST-cnts read ({sec})")

    Image.MAX_IMAGE_PIXELS = None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_uni(dev)
    img = np.asarray(Image.open(H.image_path(sec)).convert("RGB"))
    cp = man["sections"][sec]["crop_px"]
    idx = rng.choice(len(spt), size=min(8, len(spt)), replace=False)
    x = np.stack([to_uni_input(crop(img, spt["pixel_x"].iloc[i], spt["pixel_y"].iloc[i], cp),
                               man.get("grayscale", False)) for i in idx])
    with torch.inference_mode():
        f = m(torch.from_numpy(x).to(dev)).float().cpu().numpy()
    ref = data[sec]["features"][idx]
    cos = (f * ref).sum(1) / (np.linalg.norm(f, axis=1) * np.linalg.norm(ref, axis=1))
    # a mis-aligned row would pair a spot with ANOTHER spot's embedding: cos typically < 0.9
    report("PASS" if cos.min() > 0.999 else "FAIL", f"UNI features re-derived for 8 spots ({sec})",
           f"min cosine {cos.min():.5f}")

    es = EvalSet(data[sec], genes)
    item = es[0]
    ok = torch.equal(item.labels, torch.from_numpy(data[sec]["labels"]))
    report("PASS" if ok else "FAIL", "EvalSet keeps row order (constant_1.0 -> arange)")

    from stflow_import import load, build_interpolant
    import argparse
    U = load()
    a = argparse.Namespace(**C.UPSTREAM)
    a.n_genes = len(genes)
    a.device = 0 if torch.cuda.is_available() else "cpu"
    model = U["Denoiser"](a).to(a.device)
    interp = build_interpolant(a)
    ts = TrainSet([data[s] for s in sorted(data)[:4]], a.patch_distribution, a.sample_times)
    dl = torch.utils.data.DataLoader(ts, batch_size=a.batch_size, collate_fn=U["padding_batcher"]())
    b = [t.to(a.device) for t in next(iter(dl))]
    ne, tt = interp.corrupt_exp(b[2])
    _, loss = model(exp=ne, img_features=b[0], coords=b[1], labels=b[2], t_steps=tt)
    loss.backward()
    report("PASS" if torch.isfinite(loss) else "FAIL", "one real training step",
           f"batch feats {tuple(b[0].shape)}  loss {loss.item():.3f}  steps/epoch(4 secs)={len(dl)}")
    lab_all = np.concatenate([v["labels"] for v in data.values()])
    report("PASS", "target scale (log1p raw)",
           f"mean {lab_all.mean():.3f}  max {lab_all.max():.2f}  zero-frac {(lab_all == 0).mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="*", default=["env", "model", "data", "uni", "cache"])
    ap.add_argument("--cache_tag", default="uni_v1_hest112")
    a = ap.parse_args()
    print(f"paths: STFLOW_REPO={C.STFLOW_REPO}\n       HER2ST_ROOT={C.HER2ST_ROOT}\n"
          f"       PANEL={C.PANEL}\n       UNI_CKPT={C.UNI_CKPT}\n       CACHE={C.CACHE_ROOT}")
    for s in a.stage:
        print(f"\n===== stage: {s} =====")
        {"env": stage_env, "model": stage_model, "data": stage_data, "uni": stage_uni,
         "cache": lambda: stage_cache(a.cache_tag)}[s]()
    nf, nw = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print(f"\npreflight: {RESULTS.count('PASS')} PASS, {nw} WARN, {nf} FAIL")
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
