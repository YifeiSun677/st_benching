"""
Cache-backed dataset. Same interface as Her2stCLIPDataset, so train /
infer / diagnostics don't care which one they were handed.

Patches are stored unaugmented; the flip/rotate augmentation is still
applied per __getitem__ exactly as before, so training is unchanged.
"""
import json
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class CachedCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, sections, panel, is_train, verbose=True):
        with open(os.path.join(cache_dir, "index.json")) as fh:
            self.index = json.load(fh)

        if list(self.index["panel"]) != list(panel):
            raise SystemExit(
                "cache was built with a different panel -- rebuild it "
                "(build_patch_cache.py) or pass the matching panel file")

        missing = [s for s in sections if s not in self.index["sections"]]
        if missing:
            raise SystemExit(f"sections not in cache: {missing}")

        self.is_train = is_train
        self.sections = list(sections)
        self._patches = np.load(os.path.join(cache_dir, "patches.npy"),
                                mmap_mode="r")
        expr = np.load(os.path.join(cache_dir, "expression.npy"))

        rows = []
        for s in sections:
            a, b = self.index["sections"][s]
            rows.append(np.arange(a, b))
        self.rows = np.concatenate(rows) if rows else np.array([], dtype=int)
        self.expression = expr[self.rows]
        all_keys = self.index["spot_keys"]
        self.keys = [all_keys[r] for r in self.rows]

        if verbose:
            print(f"cache: {len(self.rows)} spots from {len(sections)} "
                  f"sections (is_train={is_train})")

    def expression_matrix(self):
        return self.expression

    def spot_keys(self):
        return self.keys

    def transform(self, arr):
        image = Image.fromarray(arr)
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)
            image = TF.rotate(image, random.choice([180, 90, 0, -90]))
        image = TF.to_tensor(image)
        return TF.normalize(image, mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    def __getitem__(self, i):
        arr = np.asarray(self._patches[self.rows[i]])
        return {
            "image": self.transform(arr).float(),
            "reduced_expression": torch.from_numpy(self.expression[i]).float(),
        }

    def __len__(self):
        return len(self.rows)


def build_dataset(args, sections, panel, is_train, verbose=True):
    """Pick cache or on-the-fly loader based on --cache."""
    if getattr(args, "cache", None):
        return CachedCLIPDataset(args.cache, sections, panel, is_train, verbose)
    from her2st_dataset import Her2stCLIPDataset
    return Her2stCLIPDataset(args.root, sections, panel, is_train, verbose)
