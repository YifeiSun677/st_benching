"""
Build a one-off patch cache for the whole of her2st.

Why: LOPO folds reference ~30 sections. her2st images are ~9300x9900, i.e.
~275 MB each decoded, so a fold would hold ~8.3 GB resident and re-decode
all of it three times (train, infer, diagnostics). Cropping once to
224x224 gives 13,620 x 224 x 224 x 3 = ~2.05 GB on disk, memmapped, with
no JPEG decode at run time.

Run once:
    python build_patch_cache.py --root /workspace/her2st/data \
        --panel ../panels/panel_833.txt --out /workspace/her2st_cache

Produces:
    patches.npy     (N, 224, 224, 3) uint8   -- memmapped at load
    expression.npy  (N, n_genes) float32     -- CPM natural log1p
    index.json      section -> [start, end), spot keys, panel, provenance
"""
import argparse
import glob
import json
import os

import numpy as np

from her2st_dataset import PATCH, Her2stSection, load_panel


def all_sections(root):
    hits = []
    for ext in ("tsv.gz", "tsv"):
        hits += glob.glob(os.path.join(root, "ST-cnts", f"*.{ext}"))
    return sorted({os.path.basename(p).split(".")[0] for p in hits})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    panel = load_panel(args.panel)
    sections = all_sections(args.root)
    print(f"{len(sections)} sections: {sections}")

    # pass 1: spot counts, so the memmap can be allocated exactly
    print("\nPass 1/2 -- counting spots")
    counts, meta = {}, {}
    for s in sections:
        sec = Her2stSection(args.root, s, panel, verbose=False)
        counts[s] = len(sec)
        meta[s] = {"n_spots": len(sec), "n_padded": sec.n_padded,
                   "image_hw": list(sec.image.shape[:2])}
        print(f"  {s}: {len(sec)} spots ({sec.n_padded} padded)")
        del sec

    total = sum(counts.values())
    gb = total * PATCH * PATCH * 3 / 1e9
    print(f"\ntotal {total} spots -> patches.npy will be {gb:.2f} GB")

    patches = np.lib.format.open_memmap(
        os.path.join(args.out, "patches.npy"), mode="w+",
        dtype=np.uint8, shape=(total, PATCH, PATCH, 3))
    expression = np.zeros((total, len(panel)), dtype=np.float32)

    print("\nPass 2/2 -- cropping")
    ranges, keys, cursor = {}, [], 0
    for s in sections:
        sec = Her2stSection(args.root, s, panel, verbose=False)
        n = len(sec)
        for i in range(n):
            patches[cursor + i] = sec.patch(i)
        expression[cursor:cursor + n] = sec.expression
        keys += [f"{s}:{sid}" for sid in sec.spot_ids]
        ranges[s] = [cursor, cursor + n]
        cursor += n
        print(f"  {s}: rows {ranges[s][0]}-{ranges[s][1]}")
        del sec

    assert cursor == total, (cursor, total)
    patches.flush()
    np.save(os.path.join(args.out, "expression.npy"), expression)
    with open(os.path.join(args.out, "index.json"), "w") as fh:
        json.dump({
            "sections": ranges,
            "spot_keys": keys,
            "panel": panel,
            "patch_size": PATCH,
            "n_spots": total,
            "source_root": os.path.abspath(args.root),
            "normalisation": "CPM natural log1p, no Harmony",
            "section_meta": meta,
        }, fh)

    print(f"\ncache written to {args.out} ({gb:.2f} GB)")


if __name__ == "__main__":
    main()
