"""
stflow_port/upstream_vendored.py -- two objects copied VERBATIM from upstream
stflow/data/dataset.py @ 880c2ee: `SPData` and `padding_batcher`.

Why copied instead of imported: importing upstream stflow/data/dataset.py pulls in
stflow/hest_utils/st_dataset.py, which imports scanpy, matplotlib, scikit-learn and h5py
at module level (scanpy alone drags in numba + llvmlite, ~70 MB). The port uses none of
that code. Copying these ~30 lines removes the whole chain.

preflight.py (stage `model`) re-reads the clone and asserts that the source text of both
objects is character-for-character identical to what is below.
"""
import torch
import torch.nn.functional as F

VENDORED_NAMES = ("SPData", "padding_batcher")
UPSTREAM_FILE = "stflow/data/dataset.py"


class SPData:
    features: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    coords: torch.Tensor | None = None

    def __init__(self, features, labels, coords):
        self.features = features
        self.labels = labels
        self.coords = coords

        # decenter
        self.coords[:, 0] = self.coords[:, 0] - self.coords[:, 0].mean()
        self.coords[:, 1] = self.coords[:, 1] - self.coords[:, 1].mean()

    def __len__(self):
        return len(self.features)

    def chunk(self, index):
        return SPData(
            features=self.features[index],
            labels=self.labels[index],
            coords=self.coords[index]
        )


def padding_batcher():
    def batcher_dev(batch):
        features = [d.features for d in batch]
        labels = [d.labels for d in batch]
        coords = [d.coords for d in batch]

        max_len = max([x.size(0) for x in features])
        features = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in features])
        labels = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in labels])
        coords = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in coords])

        return features, coords, labels
    return batcher_dev
