"""her2st → 768 维 CTransPath 特征缓存。一次性，之后所有 fold 复用。"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None    # her2st 图很大，关掉 decompression bomb 限制

from .config import (ST_ROOT, CTRANSPATH, PANEL_FILE, FEATURE_DIR, PATCH_PX, SEED)
from .p2s_import import (CTransPathExtractor, macenko_normalizer,
                         evaluate_tile, init_random_seed)

# ------------------------------------------------------------------ #
# her2st 布局解析（★ 若与你的 loader 不同，只改这三个函数）
# 按 almaan/ViT_HER2ST 标准布局；行=spot 'XxY'
# ------------------------------------------------------------------ #
def iter_sections():
    """产出所有 section 名，如 'A1','A2',...,'H1'。"""
    cnt_dir = ST_ROOT / "ST-cnts"
    for f in sorted(cnt_dir.glob("*.tsv*")):
        yield f.name.split(".tsv")[0]

def _read_counts(section):
    """读 counts：行=spot 'XxY'，列=基因。返回 DataFrame。"""
    cnt_dir = ST_ROOT / "ST-cnts"
    p = cnt_dir / f"{section}.tsv.gz"
    if not p.exists():
        p = cnt_dir / f"{section}.tsv"
    return pd.read_csv(p, sep="\t", index_col=0,
                       compression="gzip" if str(p).endswith(".gz") else None)

def _read_spotfile(section):
    """读 spot 选择文件：需含 array 坐标 x,y 与像素坐标 pixel_x,pixel_y。"""
    sp = ST_ROOT / "ST-spotfiles" / f"{section}_selection.tsv"
    df = pd.read_csv(sp, sep="\t")
    # 兼容不同表头命名
    ren = {}
    for a, b in [("new_x", "x"), ("new_y", "y"), ("pixel_x", "px"), ("pixel_y", "py"),
                 ("X", "x"), ("Y", "y")]:
        if a in df.columns:
            ren[a] = b
    df = df.rename(columns=ren)
    if "px" not in df.columns and "pixel_x" in df.columns:
        df = df.rename(columns={"pixel_x": "px", "pixel_y": "py"})
    df["x"] = df["x"].round().astype(int)
    df["y"] = df["y"].round().astype(int)
    df["id"] = df["x"].astype(str) + "x" + df["y"].astype(str)
    return df[["id", "x", "y", "px", "py"]]

def _read_image(section):
    letter = section[0]
    d = ST_ROOT / "ST-imgs" / letter / section
    jpgs = (list(d.glob("*.jpg")) + list(d.glob("*.jpeg"))
            + list(d.glob("*.tif")) + list(d.glob("*.png")))
    assert jpgs, f"找不到 {section} 的 H&E 图：{d}"
    return np.asarray(Image.open(jpgs[0]).convert("RGB"))   # (H,W,3) uint8
# ------------------------------------------------------------------ #

def _crop(img, px, py, r):
    """以 (px,py) 为中心裁 2r×2r，越界处 0 填充。"""
    H, W, _ = img.shape
    x0, y0, x1, y1 = px - r, py - r, px + r, py + r
    tile = np.zeros((2*r, 2*r, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    tile[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = img[sy0:sy1, sx0:sx1]
    return tile

def build(only_patients=None):
    init_random_seed(SEED)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    panel = [g.strip() for g in PANEL_FILE.read_text().splitlines() if g.strip()]
    assert len(panel) == 833, f"panel 应为 833，实际 {len(panel)}"

    ext = CTransPathExtractor(str(CTRANSPATH))     # 冻结主干，GPU 自动
    norm = macenko_normalizer()
    r = PATCH_PX // 2

    manifest = {}
    for section in iter_sections():
        if only_patients and section[0] not in only_patients:
            continue
        cnts = _read_counts(section)
        spots = _read_spotfile(section)
        # 只保留同时有 counts 和坐标的 spot，并按 counts 行顺序对齐
        spots = spots[spots["id"].isin(cnts.index)].reset_index(drop=True)
        if len(spots) == 0:
            print(f"[skip] {section}: 无对齐 spot"); continue
        img = _read_image(section)

        tiles, keep_flags = [], []
        for _, row in spots.iterrows():
            tile = _crop(img, int(row.px), int(row.py), r)
            sel = int(evaluate_tile(tile, 15, 0.5))   # 质量 flag：记录但不丢
            keep_flags.append(sel)
            try:
                normed = norm.transform(tile)          # 逐 tile Macenko → 固定 target
            except Exception:
                normed = tile                          # 归一化失败则用原图，保证行数不变
            tiles.append(Image.fromarray(normed))

        feats = ext.extract(tiles)                     # (n,768) float32
        # counts 对齐到 833 panel，缺失基因 0 填充（你的既定约定）
        c = cnts.loc[spots["id"].values]
        c = c.reindex(columns=panel, fill_value=0).to_numpy(dtype=np.float32)

        out = FEATURE_DIR / f"{section}.npz"
        np.savez_compressed(
            out,
            feat=feats.astype(np.float32),
            counts833=c,
            spot_id=spots["id"].to_numpy(),
            ax=spots["x"].to_numpy(np.int32),
            ay=spots["y"].to_numpy(np.int32),
            select=np.asarray(keep_flags, np.int8),
            genes=np.asarray(panel),
        )
        manifest[section] = {"patient": section[0], "n_spots": int(len(spots)),
                             "n_pass_qc": int(sum(keep_flags))}
        print(f"[ok] {section}: {len(spots)} spots, QC pass {sum(keep_flags)}")

    (FEATURE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n特征缓存完成 → {FEATURE_DIR}  （{len(manifest)} sections）")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="*", default=None,
                    help="只对这些 patient 建特征（测试用），如 --patients A B")
    a = ap.parse_args()
    build(only_patients=a.patients)
