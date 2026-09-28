"""
Drop-in replacement for HisToGene's ViT_HER2ST, reading the patch cache.

Three deliberate differences from the original, all required by the benchmark:

1. LOPO, not leave-one-section-out. The original does `samples = names[1:33]`,
   which is leave-one-SECTION-out AND silently drops A1 and all of patient H.
   Here fold 0..7 holds out patients A..H and all 36 sections are used.
2. Gene panel is a real argument. The original ignores its own `gene_list`
   parameter and hardcodes data/her_hvg_cut_1000.npy (785 genes).
3. Genes missing from a section's count matrix are zero-filled, so the panel
   width is fixed across folds and models.

Everything else is kept identical to the original on purpose: no stain
normalization, no augmentation (the original defines self.transforms but never
calls it in the ViT path), and raw 0-255 pixel values fed straight into the
patch embedding with no rescaling.
"""
import os

import numpy as np
import scprep as scp
import torch

PATIENTS = list("ABCDEFGH")


def load_panel(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


class CachedHER2ST(torch.utils.data.Dataset):
    """One item == one whole section, matching the original's batching."""

    def __init__(self, train=True, fold=0, gene_list=None, cache="cache"):
        if gene_list is None:
            raise ValueError("gene_list is required")
        self.gene_list = list(gene_list)
        self.train = train
        self.fold = fold

        all_names = sorted(f[:-4] for f in os.listdir(cache) if f.endswith(".npz"))
        if not all_names:
            raise SystemExit(f"empty cache at {cache}/ -- run build_cache.py first")
        held_out = PATIENTS[fold]
        self.names = [n for n in all_names if (n[0] != held_out) == train]

        self.items = []
        for n in self.names:
            d = np.load(f"{cache}/{n}.npz", allow_pickle=True)

            idx = {g: i for i, g in enumerate(d["genes"])}
            cols = np.array([idx.get(g, -1) for g in self.gene_list])
            ok = cols >= 0
            m = np.zeros((d["counts"].shape[0], len(self.gene_list)), dtype=np.float32)
            m[:, ok] = d["counts"][:, cols[ok]]  # zero-fill for undetected genes

            exp = scp.transform.log(scp.normalize.library_size_normalize(m))
            self.items.append((
                torch.from_numpy(d["patches"]),                             # uint8
                torch.from_numpy(d["loc"]).long(),
                torch.from_numpy(np.asarray(exp, dtype=np.float32)),
                torch.from_numpy(d["centers"]).float(),
            ))

        # position embeddings are nn.Embedding(n_pos, dim); array coords must fit
        mx = max(int(it[1].max()) for it in self.items)
        if mx >= 64:
            raise ValueError(f"max array coord {mx} >= n_pos=64; raise n_pos")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        patches, loc, exp, centers = self.items[i]
        patches = patches.float()  # uint8 -> float32 here, keeps RAM 4x smaller
        if self.train:
            return patches, loc, exp
        return patches, loc, exp, centers

    def summary(self):
        spots = sum(len(it[1]) for it in self.items)
        mb = sum(it[0].numel() for it in self.items) / 1e6
        return (f"fold {self.fold} (holds out {PATIENTS[self.fold]}) | "
                f"{'train' if self.train else 'test'} | {len(self.names)} sections | "
                f"{spots} spots | {len(self.gene_list)} genes | {mb:.0f} MB patches")
