"""Pre-flight checks + a 2-epoch smoke test on 3 sections.

    python -m histogene.preflight

Everything should say OK before you launch a real fold.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import cache, config as C, her2st
from .dataset import HER2STSections

ok = True


def check(label: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"[{'OK ' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def note(label: str, cond: bool, detail: str = "") -> None:
    """Informational: printed, but does not fail the pre-flight."""
    print(f"[{'OK ' if cond else 'WARN'}] {label}" + (f"  {detail}" if detail else ""))


def main() -> None:
    print("=== paths ===")
    print(f"       repo root: {C.REPO_ROOT}")
    check("her2st ST-cnts", C.CNT_DIR.is_dir(), str(C.CNT_DIR))
    check("her2st ST-imgs", C.IMG_DIR.is_dir(), str(C.IMG_DIR))
    check("her2st ST-spotfiles", C.POS_DIR.is_dir(), str(C.POS_DIR))
    check("HisToGene repo", (C.HISTOGENE_REPO / "vis_model.py").exists(),
          str(C.HISTOGENE_REPO))
    check("panel file", Path(C.PANEL_FILE).exists(), str(C.PANEL_FILE))
    note("benchmark gene sets", C.GENE_SETS_DIR.is_dir(), str(C.GENE_SETS_DIR))
    if not ok:
        sys.exit("fix the paths above first")

    print("\n=== imports ===")
    sys.path.insert(0, str(C.HISTOGENE_REPO))
    import pytorch_lightning as pl
    from vis_model import HisToGene
    print(f"       torch {torch.__version__}  lightning {pl.__version__}")
    check("CUDA available", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

    print("\n=== data ===")
    names = her2st.section_names()
    note("36 sections", len(names) == 36, f"found {len(names)}: {names[:4]}...")
    panel = her2st.load_panel()
    note("833-gene panel", len(panel) == 833, f"{len(panel)} genes")

    missing_cache = [n for n in names
                     if not (cache.patch_dir() / f"{n}.npy").exists()]
    check("patch cache complete", not missing_cache,
          "missing: " + ", ".join(missing_cache[:5]) if missing_cache else "")
    if missing_cache:
        sys.exit("run  python -m histogene.build_cache  first")

    print("\n=== target arithmetic ===")
    meta = her2st.read_meta(names[0])
    X = her2st.expression(meta, panel)
    check("expression shape", X.shape == (len(meta), len(panel)), str(X.shape))
    check("target is log10(CP10K+1)", X.min() >= 0 and X.max() < 5,
          f"min {X.min():.3f} max {X.max():.3f}")
    try:
        import scprep
        raw = np.column_stack([meta[g].values if g in meta.columns
                               else np.zeros(len(meta)) for g in panel]).astype(float)
        ref = scprep.transform.log(scprep.normalize.library_size_normalize(raw))
        check("matches scprep", np.allclose(X, ref, atol=1e-4))
    except Exception as e:
        print(f"[skip] scprep cross-check ({type(e).__name__}) - not required")

    print("\n=== folds ===")
    for cv in ("patient", "section"):
        f = her2st.folds(cv)
        print(f"       cv={cv:8s} {len(f):2d} folds, fold0 holds out {f[0]['test']}")

    print("\n=== one forward pass ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = HER2STSections(names[:3], panel, train=True)
    patches, positions, exps = ds[0]
    check("patch dim", patches.shape[1] == C.PATCH_DIM,
          f"{tuple(patches.shape)} expected [n, {C.PATCH_DIM}]")
    check("patch scale is 0-255", patches.max() > 1.5,
          f"max {patches.max():.1f} (the repo does NOT divide by 255)")
    check("array coords < n_pos", int(positions.max()) < C.N_POS,
          f"max {int(positions.max())}")

    model = HisToGene(patch_size=C.PATCH_SIZE, n_layers=C.N_LAYERS,
                      n_genes=len(panel), dim=C.DIM, learning_rate=C.LR,
                      dropout=C.DROPOUT, n_pos=C.N_POS).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"       parameters: {n_par/1e6:.1f} M")
    with torch.no_grad():
        y = model(patches.unsqueeze(0).to(dev), positions.unsqueeze(0).to(dev))
    check("output shape", tuple(y.shape) == (1, patches.shape[0], len(panel)),
          str(tuple(y.shape)))

    print("\n=== 2-epoch smoke test (3 sections) ===")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, max_epochs=2, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         log_every_n_steps=1, precision="32-true")
    t0 = time.time()
    trainer.fit(model, loader)
    dt = time.time() - t0
    per_section = dt / (2 * len(ds))
    print(f"       2 epochs x 3 sections in {dt:.1f}s  ->  {per_section:.2f} s/section")
    print(f"       estimate for one 100-epoch fold (31 train sections): "
          f"{per_section*31*100/60:.0f} min")
    if torch.cuda.is_available():
        print(f"       peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
