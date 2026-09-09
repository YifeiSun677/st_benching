"""
Datasets that reproduce upstream TriDataset's contract.

TRAIN (per-spot):
  returns dict img (3,224,224), mask (25,), neighbor_emb (25,512),
  label (833,), pid (scalar), sid (scalar) -- collated to 1-D [B]. The dataset
  also exposes int2id / global_embs
  / pos_dict, which the model's retrieve_global_emb() reads to encode the whole
  held-in section's global tokens each step -- so training MUST pass
  `dataset=<this>` in the forward call (train.py does).

TEST (per-section): section_batch() returns the tensors for one held-out
  section in one shot (img (N,3,224,224), mask (N,25), neighbor_emb (N,25,512),
  global_emb (1,N,512), position (N,2), label (N,833)). Fed straight to the
  model's inference path -- no DataLoader, so no stray batch dim to squeeze.
"""
import os
import numpy as np
import h5py
import torch
import torchvision.transforms as T

from . import config, her2st

_NORM = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
_TRAIN_TF = T.Compose([
    T.ToPILImage(),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation((90, 90))]),
    T.ToTensor(), _NORM,
])
_TEST_TF = T.Compose([T.ToTensor(), _NORM])


def _open_cache():
    # patches.npy is a real .npy (header + data); np.load with mmap_mode reads
    # the header and returns a correctly-shaped read-only memmap.
    arr = np.load(config.HER2ST_CACHE, mmap_mode="r")
    if arr.shape[1:] != config.HER2ST_CACHE_SHAPE[1:]:
        raise ValueError(f"cache patch shape {arr.shape[1:]} != "
                         f"{config.HER2ST_CACHE_SHAPE[1:]}")
    return arr


def _load_section(section, panel):
    """Read one section's feature file + build its expression matrix."""
    with h5py.File(os.path.join(config.FEATURE_DIR, f"{section}.h5"), "r") as f:
        d = dict(
            features=f["features"][:],            # (N,512) global
            coords=f["coords"][:],                # (N,2) (x,y)
            neighbor=f["neighbor"][:],            # (N,25,512)
            mask=f["mask"][:],                    # (N,25)
            cache_idx=f["cache_idx"][:],          # (N,)
            array_rc=f["array_rc"][:],            # (N,2) (row,col)
            spot_id=f["spot_id"][:].astype(str),  # (N,)
        )
    counts = her2st.load_counts(section, panel, d["spot_id"])
    d["expr"] = her2st.normalize_expr(counts, d["array_rc"])   # (N,833)
    return d


class TriTrainDataset(torch.utils.data.Dataset):
    def __init__(self, sections, panel):
        self.panel = panel
        self.ids = list(sections)
        self.int2id = dict(enumerate(self.ids))
        self.cache = _open_cache()

        self.sec = {s: _load_section(s, panel) for s in self.ids}
        self.lengths = [len(self.sec[s]["expr"]) for s in self.ids]
        self.cumlen = np.cumsum(self.lengths)

        # what the model's retrieve_global_emb() consumes
        self.global_embs = {s: torch.FloatTensor(self.sec[s]["features"])
                            for s in self.ids}
        self.pos_dict = {s: torch.FloatTensor(self.sec[s]["coords"])
                         for s in self.ids}

    def __len__(self):
        return int(self.cumlen[-1])

    def __getitem__(self, index):
        i = int(np.searchsorted(self.cumlen, index, side="right"))
        idx = index - (self.cumlen[i - 1] if i > 0 else 0)
        s = self.int2id[i]
        d = self.sec[s]

        patch = np.asarray(self.cache[int(d["cache_idx"][idx])])   # (224,224,3)
        img = _TRAIN_TF(patch)
        return {
            "img": img,
            "mask": torch.LongTensor(d["mask"][idx]),
            "neighbor_emb": torch.FloatTensor(d["neighbor"][idx]),
            "label": torch.FloatTensor(d["expr"][idx]),
            # scalars -> default_collate stacks them into 1-D [B], which is what
            # the model needs: pid is used as a row-mask (pid == section_id) and
            # sid.shape[0] must equal the batch size. Returning [1] here would
            # collate to [B,1] and break encode_global's masked assignment.
            "pid": torch.tensor(i, dtype=torch.long),          # section int id
            "sid": torch.tensor(int(idx), dtype=torch.long),   # row within section
        }


class TriTestSections:
    """Held-out patient's sections, served one section at a time for inference."""
    def __init__(self, sections, panel):
        self.panel = panel
        self.ids = list(sections)
        self.cache = _open_cache()
        self.sec = {s: _load_section(s, panel) for s in self.ids}

    def section_batch(self, section, device="cuda"):
        d = self.sec[section]
        patches = np.asarray(self.cache[d["cache_idx"].astype(int)])
        img = torch.stack([_TEST_TF(p) for p in patches], 0).to(device)  # (N,3,224,224)
        mask = torch.LongTensor(d["mask"]).to(device)                    # (N,25)
        neighbor = torch.FloatTensor(d["neighbor"]).to(device)          # (N,25,512)
        global_emb = torch.FloatTensor(d["features"]).unsqueeze(0).to(device)  # (1,N,512)
        position = torch.FloatTensor(d["coords"]).to(device)            # (N,2)
        label = d["expr"]                                                # (N,833) numpy
        return dict(img=img, mask=mask, neighbor_emb=neighbor,
                    global_emb=global_emb, position=position,
                    label=label, spot_id=d["spot_id"], coords=d["coords"])
