import sys
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from . import config as C
from .her2st import load_section, load_panel

sys.path.insert(0, str(C.HIST2ST_REPO))
from graph_construction import calcADJ          # noqa: E402  (original repo, Grid pruning)


class Her2stHist2ST(Dataset):
    """One item = one section. Returns (patch, position, exp, adj, ori, sf, center),
    same order as biomed-AI/Hist2ST's ViT_HER2ST(ori=True, adj=True, flatten=False)."""

    def __init__(self, sections, neighs=4, prune="Grid", scale255=False):
        self.sections = list(sections)
        self.panel = load_panel()
        self.scale255 = scale255
        self.patches = np.load(C.CACHE_DIR / "patches.npy", mmap_mode="r")
        idx = pd.read_csv(C.CACHE_DIR / "index.csv")
        self.rows = {s: g.sort_values("row") for s, g in idx.groupby("section")}

        self.data, self.adj = {}, {}
        for s in self.sections:
            d = load_section(s, self.panel)
            r = self.rows[s]
            assert list(r.spot_id.astype(str)) == list(d["spot_id"]), \
                f"{s}: cache and loader row order disagree"
            d["row"] = r.row.values
            self.data[s] = d
            self.adj[s] = calcADJ(d["array"], neighs, pruneTag=prune)   # Grid prune is in grid units

    def __len__(self):
        return len(self.sections)

    def __getitem__(self, i):
        s = self.sections[i]
        d = self.data[s]
        p = np.asarray(self.patches[d["row"]])                   # [N,112,112,3] uint8
        patch = torch.from_numpy(p).permute(0, 3, 2, 1).float()   
        if self.scale255:
            patch = patch / 255.0
        return (patch,
                torch.from_numpy(d["array"]),
                torch.from_numpy(d["exp"]),
                self.adj[s].float(),
                torch.from_numpy(d["ori"]),
                torch.from_numpy(d["sf"]),
                torch.from_numpy(d["pixel"]))
