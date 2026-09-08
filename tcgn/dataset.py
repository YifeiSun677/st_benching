"""
Per-spot torch Dataset for the TCGN port.

Yields (image[3,224,224] float, expr[833] float). Images come from the uint8
112x112 cache, then follow the upstream transform chain exactly:
    patch/255 -> Resize(224, antialias) -> ImageNet-normalise
plus, for TRAIN only, RandomRotation(180)+H/V flips (upstream train_transform).

We assemble per-fold arrays in memory (as upstream's ST_HER2_Dataset does) by
slicing the global cache for the fold's sections. Targets are read fresh from
her2_data so the 833-panel / zero-fill / target-scale logic stays in one place.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import config as C
import her2_data as H
import patch_cache as PC


def _basic_transform():
    return T.Compose([
        T.Resize((C.RESIZE, C.RESIZE), antialias=True),
        T.ConvertImageDtype(torch.float),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def _train_aug():
    return T.Compose([
        T.RandomRotation(180),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
    ])


class TCGNDataset(Dataset):
    def __init__(self, sections, train):
        self.train = train
        self.basic = _basic_transform()
        self.aug = _train_aug()

        patches, sec_arr, sid_arr, _ = PC.load()
        panel = H.load_panel()

        imgs, exps, meta_sec, meta_sid = [], [], [], []
        for name in sections:
            rows = np.where(sec_arr == name)[0]
            if len(rows) == 0:
                raise RuntimeError("section %s not in patch cache; rebuild cache" % name)
            expr, _, spot_ids = H.section_targets(name, panel)
            # Cache and target rows must correspond 1:1 and in the same order.
            assert len(rows) == len(spot_ids), \
                "%s: cache has %d spots, targets %d" % (name, len(rows), len(spot_ids))
            cache_sids = sid_arr[rows].astype(str)
            if not np.array_equal(cache_sids, spot_ids.astype(str)):
                # reorder cache rows to match the target spot order
                order = {s: j for j, s in enumerate(cache_sids)}
                perm = np.array([order[s] for s in spot_ids.astype(str)])
                rows = rows[perm]
            # HWC uint8 -> CHW uint8 tensor, then /255 (upstream divides by 255
            # before the basic transform's ConvertImageDtype no-op on floats).
            blk = np.asarray(patches[rows])                      # (n,112,112,3)
            t = torch.from_numpy(blk).permute(0, 3, 1, 2).contiguous().float() / 255.0
            t = self.basic(t)                                    # (n,3,224,224)
            imgs.append(t)
            exps.append(torch.from_numpy(expr))
            meta_sec += [name] * len(spot_ids)
            meta_sid += list(spot_ids.astype(str))

        self.imgs = torch.cat(imgs, dim=0)
        self.exps = torch.cat(exps, dim=0)
        self.section = np.asarray(meta_sec)
        self.spot_id = np.asarray(meta_sid)

    def __len__(self):
        return self.exps.shape[0]

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.train:
            img = self.aug(img)
        return img, self.exps[i]
