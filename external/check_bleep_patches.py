#!/usr/bin/env python
"""Diagnostic: are the on-the-fly 224 px crops pixel-identical to the her2st_cache patches
the original BLEEP LOPO run used?  Tells whether the round-trip wobble comes from the crop
path or from GPU non-determinism in the image encoder.

usage: cd /workspace/st_benching && python external/check_bleep_patches.py B1
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_bleep  # noqa: F401,E402  (applies the .tsv/.tsv.gz read_counts patch)
import common as K  # noqa: E402
from her2st_dataset import Her2stSection  # noqa: E402

sec_name = sys.argv[1] if len(sys.argv) > 1 else "B1"
cache = "/workspace/her2st_cache"
idx = json.load(open(os.path.join(cache, "index.json")))
P = np.load(os.path.join(cache, "patches.npy"), mmap_mode="r")
a, b = idx["sections"][sec_name]
keys = [str(k) for k in idx["spot_keys"][a:b]]
row = {k: i for i, k in enumerate(keys)}
sec = Her2stSection(str(K.HER2ST_ROOT), sec_name, K.load_panel(), verbose=False)
diffs, missing = [], 0
for i, s in enumerate(sec.spot_ids):
    k = f"{sec_name}:{s}" if f"{sec_name}:{s}" in row else s
    if k not in row:
        missing += 1
        continue
    c = np.asarray(P[a + row[k]]).astype(int)
    o = sec.patch(i).astype(int)
    diffs.append(np.abs(c - o).max())
d = np.array(diffs)
print(f"{sec_name}: {len(d)} spots compared, {missing} not matched by key")
print(f"pixel-identical: {100*(d == 0).mean():.1f}%   max |diff|: {d.max()}   median: {np.median(d)}")
print("order in cache == order on the fly:", keys == [f"{sec_name}:{s}" for s in sec.spot_ids] or keys == list(sec.spot_ids))
