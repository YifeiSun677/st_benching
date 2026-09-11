"""
stflow_port/dataset.py -- her2st datasets that reproduce upstream STFlow sampling exactly.

The containers are upstream's own `SPData` (decentres coords per section),
`PatchSampler` and `padding_batcher`, imported from the clone, so the training
distribution is byte-for-byte the released one:

  TrainSet  == upstream MultiHESTDataset: each epoch draws `sample_times` items per
            training section; each item is a random spatially-contiguous sub-patch
            (KD-tree nearest neighbours of a random centre) whose size is
            max(2, int(N * U(0,1)))  ('uniform' patch_distribution).
  EvalSet   == upstream HESTDataset(distribution='constant_1.0', sample_times=1):
            the whole section, in stored row order (PatchSampler returns arange when
            the patch is the full section - asserted in preflight).
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C
from stflow_import import load


def load_cache(tag, sections=None):
    d = C.cache_dir(tag)
    man = json.load(open(os.path.join(d, "manifest.json")))
    genes = list(man["genes"])
    secs = sections or sorted(man["sections"].keys())
    out = {}
    for s in secs:
        z = np.load(os.path.join(d, f"{s}.npz"), allow_pickle=False)
        g = [str(x) for x in z["genes"]]
        if g != genes:
            raise ValueError(f"{s}: gene order differs from manifest")
        out[s] = dict(section=s, patient=s[0], features=z["features"], labels=z["labels"],
                      coords=z["coords"], spot_id=z["spot_id"].astype(str))
    return out, genes, man


def split_sections(all_sections, test_patient=None, val_patient=None):
    test = [s for s in all_sections if s[0] == test_patient] if test_patient else []
    val = [s for s in all_sections if s[0] == val_patient] if val_patient else []
    train = [s for s in all_sections if s not in test and s not in val]
    return train, val, test


def _spdata(d):
    U = load()
    return U["SPData"](features=torch.from_numpy(d["features"]).float(),
                       labels=torch.from_numpy(d["labels"]).float(),
                       coords=torch.from_numpy(d["coords"].copy()).float())


class TrainSet(Dataset):
    def __init__(self, sec_dicts, distribution="uniform", sample_times=10):
        U = load()
        self.sp_datasets = [_spdata(d) for d in sec_dicts]
        self.n_chunks = [sample_times] * len(sec_dicts)
        self.patch_sampler = U["PatchSampler"](distribution)

    def __len__(self):
        return sum(self.n_chunks)

    def __getitem__(self, idx):                      # verbatim upstream logic
        for i, n_chunk in enumerate(self.n_chunks):
            if idx < n_chunk:
                return self.sp_datasets[i].chunk(self.patch_sampler(self.sp_datasets[i].coords))
            idx -= n_chunk


class EvalSet(Dataset):
    def __init__(self, d, gene_list, shuffle_features_seed=None):
        U = load()
        d = dict(d)
        if shuffle_features_seed is not None:        # perturbation null: break image<->spot pairing
            rng = np.random.default_rng(shuffle_features_seed)
            d["features"] = d["features"][rng.permutation(len(d["features"]))]
        self.name = d["section"]
        self.gene_list = gene_list
        self.sp_dataset = _spdata(d)
        self.patch_sampler = U["PatchSampler"]("constant_1.0")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sp_dataset.chunk(self.patch_sampler(self.sp_dataset.coords))
