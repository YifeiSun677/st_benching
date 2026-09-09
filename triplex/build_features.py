"""
One-time feature build. For each her2st section, from the shared 224px patch
cache we produce:

  global  : (N, 512)      CIGAR features of every spot           -> global branch
  coords  : (N, 2)        (array_col, array_row) = (x, y)        -> APEG position
  neighbor: (N, 25, 512)  5x5 grid of neighbouring spots' CIGAR features
  mask    : (N, 25)       1 where a neighbour spot exists, else 0
  cache_idx, spot_id, array_rc for bookkeeping / target-patch lookup at train time

Neighbour construction (documented deviation): we gather the CIGAR features of
the actual adjacent SPOTS at array offsets (-2..+2, -2..+2), centre = the target
spot at token index 12. Upstream instead re-tiles a contiguous 1120px image
region into 5x5. For grid-based her2st the spot-gather is the natural analogue
and keeps every feature on the same patches as the rest of the benchmark. The
5x5 ordering matches the neighbour encoder's attention-bias grid exactly
(itertools.product(range(5), range(5))).

Run once; costs ~ one CIGAR forward pass over 13,620 patches (a few minutes on
a 4090). Output is re-derivable, so it lives on the volume, not in git.
"""
import os
import itertools
import numpy as np
import h5py
import torch
from tqdm import tqdm

from . import config, her2st
from .cigar import build_cigar_encoder, encode_patches

# 5x5 offsets in the EXACT order the attention bias expects; centre at idx 12.
_OFFSETS = [(a - 2, b - 2) for (a, b) in itertools.product(range(5), range(5))]
assert _OFFSETS[12] == (0, 0)


def _build_neighbor(global_feat, array_rc):
    """global_feat (N,512), array_rc (N,2)=(row,col) -> (N,25,512), (N,25)."""
    n, d = global_feat.shape
    pos2idx = {(int(r), int(c)): i for i, (r, c) in enumerate(array_rc)}
    neigh = np.zeros((n, 25, d), dtype=np.float32)
    mask = np.zeros((n, 25), dtype=np.int64)
    for i, (r, c) in enumerate(array_rc):
        r, c = int(r), int(c)
        for k, (dr, dc) in enumerate(_OFFSETS):
            j = pos2idx.get((r + dr, c + dc))
            if j is not None:
                neigh[i, k] = global_feat[j]
                mask[i, k] = 1
    return neigh, mask


def build_section(section, index_df, encoder, cache, device="cuda"):
    out_path = os.path.join(config.FEATURE_DIR, f"{section}.h5")
    if os.path.exists(out_path):
        print(f"[build] {section}: exists, skipping")
        return
    rows = index_df[index_df.section == section].sort_values("idx")
    cache_idx = rows.cache_idx.values.astype(int)
    spot_id = rows.spot_id.astype(str).values
    array_rc = rows[["array_row", "array_col"]].values.astype(int)   # (row, col)
    coords_xy = rows[["array_col", "array_row"]].values.astype(np.float32)  # (x, y)

    patches = np.asarray(cache[cache_idx])                # (N,224,224,3) uint8
    global_feat = encode_patches(encoder, patches, device=device)  # (N,512)
    neigh, mask = _build_neighbor(global_feat, array_rc)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("features", data=global_feat)          # global branch
        f.create_dataset("coords", data=coords_xy)              # (x,y) for APEG
        f.create_dataset("neighbor", data=neigh)                # (N,25,512)
        f.create_dataset("mask", data=mask)                     # (N,25)
        f.create_dataset("cache_idx", data=cache_idx)
        f.create_dataset("array_rc", data=array_rc)
        f.create_dataset("spot_id", data=np.array(spot_id, dtype="S16"))
    print(f"[build] {section}: {len(cache_idx)} spots, "
          f"neighbours/spot={mask.sum(1).mean():.1f}")


def main():
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_df = her2st.load_spot_index()
    cache = np.memmap(config.HER2ST_CACHE, dtype=np.uint8, mode="r",
                      shape=config.HER2ST_CACHE_SHAPE)
    encoder = build_cigar_encoder(device=device)
    for section in tqdm(her2st.all_sections(index_df), desc="sections"):
        build_section(section, index_df, encoder, cache, device=device)
    print("[build] done ->", config.FEATURE_DIR)


if __name__ == "__main__":
    main()
