"""Build the 112x112 patch cache once: memmap uint8 [Ntotal,112,112,3] + index.csv.

If your histogene 112 cache uses the same layout, point HIST2ST_CACHE at it and
skip this script entirely.
"""
import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from . import config as C
from .her2st import list_sections, load_section, load_panel

Image.MAX_IMAGE_PIXELS = None


def img_path(name):
    d = C.IMG_DIR / name[0] / name
    return d / sorted(p.name for p in d.iterdir())[0]


def main():
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    secs = list_sections()

    meta, total = [], 0
    for s in secs:
        d = load_section(s, panel)
        for k in range(len(d["spot_id"])):
            meta.append((s, d["spot_id"][k], d["pixel"][k, 0], d["pixel"][k, 1], total + k))
        total += len(d["spot_id"])
    idx = pd.DataFrame(meta, columns=["section", "spot_id", "px", "py", "row"])
    idx.to_csv(C.CACHE_DIR / "index.csv", index=False)

    mm = np.lib.format.open_memmap(
        C.CACHE_DIR / "patches.npy", mode="w+", dtype=np.uint8,
        shape=(total, C.PATCH, C.PATCH, 3))

    for s in tqdm(secs, desc="sections"):
        im = np.asarray(Image.open(img_path(s)).convert("RGB"))
        H, W, _ = im.shape
        sub = idx[idx.section == s]
        for _, r in sub.iterrows():
            x, y = int(r.px), int(r.py)
            x0, y0 = max(0, x - C.R), max(0, y - C.R)
            x1, y1 = min(W, x + C.R), min(H, y + C.R)
            crop = im[y0:y1, x0:x1]
            buf = np.zeros((C.PATCH, C.PATCH, 3), np.uint8)
            buf[: crop.shape[0], : crop.shape[1]] = crop      # zero-pad at the border (rare)
            mm[int(r.row)] = buf
        del im
    mm.flush()
    print("cache:", total, "spots ->", C.CACHE_DIR)


if __name__ == "__main__":
    main()
