"""
Fast sanity checks before committing a GPU-hour. Run:  python preflight.py
Exits non-zero on the first failure with a clear message.
"""
import os
import sys
import numpy as np

import config as C
import her2_data as H


def check(name, cond, detail=""):
    status = "OK " if cond else "FAIL"
    print("[%s] %s %s" % (status, name, detail))
    if not cond:
        sys.exit(1)


def main():
    # 1. paths
    check("her2st ST-cnts", os.path.isdir(C.CNT_DIR), C.CNT_DIR)
    check("her2st ST-imgs", os.path.isdir(C.IMG_DIR), C.IMG_DIR)
    check("her2st ST-spotfiles", os.path.isdir(C.POS_DIR), C.POS_DIR)
    check("panel file", os.path.exists(C.PANEL_FILE), C.PANEL_FILE)

    # 2. panel + sections + folds
    panel = H.load_panel()
    check("panel length 833", len(panel) == 833, "got %d" % len(panel))
    secs = H.list_sections()
    check("sections found", len(secs) >= 32, "%d sections: %s" % (len(secs), secs))
    folds = H.lopo_folds()
    patients = [p for p, _, _ in folds]
    check("LOPO patients", len(folds) >= 1, "patients: %s" % patients)

    # 3. one section: target build, panel align, zero-fill, scale
    name = folds[0][1][0]
    expr, centers, sids = H.section_targets(name, panel)
    check("target shape [N,833]", expr.shape[1] == 833, "%s -> %s" % (name, expr.shape))
    check("targets finite", np.isfinite(expr).all())
    check("targets non-negative", (expr >= 0).all(), "log10(x+1) must be >=0")
    n_missing = 833 - H.get_meta(name).reindex(columns=panel).notna().any(axis=0).sum()
    print("     %s: %d/833 panel genes absent (zero-filled)" % (name, int(n_missing)))
    print("     target scale = %s ; example max = %.3f (CP10K~4.0, median~1-2)"
          % (C.TARGET_RESCALE, float(expr.max())))

    # 4. patch crop reproduces upstream's transpose convention (self-consistency)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import patch_cache as PC
    arr = np.asarray(Image.open(H.get_img_path(name)).convert("RGB"), dtype=np.uint8)
    cx, cy = int(centers[0][0]), int(centers[0][1])
    up = PC._crop_transposed(arr, cx, cy, C.R)                    # our reproduction
    nat = arr[cy - C.R:cy + C.R, cx - C.R:cx + C.R, :]            # natural crop
    if nat.shape == up.shape:
        rel = np.transpose(nat, (1, 0, 2))                        # natural^T
        check("patch == natural^T (upstream convention)",
              np.array_equal(up, rel), "max|diff|=%d" % int(np.abs(up.astype(int) - rel.astype(int)).max()))

    # 5. model imports + forward. Upstream hardcodes .cuda() inside the
    #    attention (transformer_block) and graph (censnet_block) modules, so the
    #    model only runs with everything on the GPU. Forward on cuda when it is
    #    available; otherwise check import + param count only (a CPU forward is
    #    impossible without editing the read-only clone).
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tcgn_import import load_tcgn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_tcgn(num_classes=833, load_cmt=False, device=dev)
    nparam = sum(p.numel() for p in m.parameters())
    check("TCGN imports + builds", nparam > 0,
          "params = %.3fM (paper ~86.24M)" % (nparam / 1e6))
    if dev == "cuda":
        m.eval()
        with torch.no_grad():
            y = m(torch.randn(2, 3, 224, 224, device=dev))
        check("TCGN forward -> [2,833]", tuple(y.shape) == (2, 833), str(tuple(y.shape)))
    else:
        print("     [note] no GPU visible; skipping the forward pass. Upstream "
              "hardcodes .cuda() in its attention/graph blocks, so a CPU forward "
              "is impossible without editing the clone. It runs on the pod's GPU.")

    print("\nAll preflight checks passed.")


if __name__ == "__main__":
    main()
