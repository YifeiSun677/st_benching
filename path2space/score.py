"""per-fold per-gene 打分：PCC / SSE ratio(vs per-section 均值) / frac_beat / markers。
   与你其他模型完全同口径。可选 Path2Space 空间平滑变体。"""
import argparse
import numpy as np
import pandas as pd
from .config import OUT_DIR, MARKERS, SMOOTH_RADIUS
from .p2s_import import smooth_genes_kdtree

def per_gene_pcc(pred, tru):
    p = pred - pred.mean(0); t = tru - tru.mean(0)
    num = (p*t).sum(0); den = np.sqrt((p**2).sum(0) * (t**2).sum(0))
    out = np.full(pred.shape[1], np.nan); m = den > 0; out[m] = num[m]/den[m]
    return out

def sse_ratio(pred, tru):
    base = np.tile(tru.mean(0), (tru.shape[0], 1))      # per-section 均值 = L2 最优常数
    sm = ((pred-tru)**2).sum(0); sb = ((base-tru)**2).sum(0)
    r = np.full(pred.shape[1], np.nan); m = sb > 0; r[m] = sm[m]/sb[m]
    return r

def _smooth(mat, ax, ay, genes):
    df = pd.DataFrame(mat, columns=list(genes))
    df["grid_x"] = ax; df["grid_y"] = ay
    sm = smooth_genes_kdtree(df, list(genes), radius=SMOOTH_RADIUS,
                             coord_cols=("grid_x", "grid_y"))
    return sm[list(genes)].to_numpy()

def score(tag="path2space_lopo_833", smooth=False):
    d = OUT_DIR.parent / tag / "preds"
    files = sorted(d.glob("*.npz"))
    assert files, f"没有预测文件：{d}"
    genes = np.load(files[0], allow_pickle=True)["genes"]
    gidx = {g: i for i, g in enumerate(genes)}

    rows_gene, per_patient = [], {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        pred, tru = z["pred"], z["truth"]
        if smooth:
            pred = _smooth(pred, z["ax"], z["ay"], genes)
            tru  = _smooth(tru,  z["ax"], z["ay"], genes)
        pcc = per_gene_pcc(pred, tru)
        r   = sse_ratio(pred, tru)
        rows_gene.append(pcc)
        p = str(z["patient"])
        per_patient.setdefault(p, []).append((pcc, r))

    G = np.vstack(rows_gene)                 # (n_section, 833)
    gene_mean = np.nanmean(G, axis=0)
    print(f"\n=== {tag}{' (smoothed)' if smooth else ''} ===")
    print(f"per-gene PCC: median {np.nanmedian(gene_mean):+.4f}  mean {np.nanmean(gene_mean):+.4f}")
    print(f"frac genes positive: {(gene_mean > 0).mean():.3f}")
    # markers
    for mk in MARKERS:
        if mk in gidx:
            print(f"  {mk:6s} {np.nanmean(G[:, gidx[mk]]):+.4f}")
    # per-patient
    print("per-patient per-gene PCC:")
    for p in sorted(per_patient):
        pccs = np.vstack([x[0] for x in per_patient[p]])
        rr   = np.vstack([x[1] for x in per_patient[p]])
        print(f"  {p}: PCC {np.nanmean(pccs):+.4f} | median SSE ratio "
              f"{np.nanmedian(rr):.3f} | frac_beat {(rr < 1).mean():.3f}")

    # 存表（与 results/<model>_<panel>/ 约定一致）
    outdir = OUT_DIR.parent / tag; suff = "_smooth" if smooth else ""
    pd.DataFrame({"gene": genes, "pcc": gene_mean}).to_csv(
        outdir / f"per_gene_pcc{suff}.csv", index=False)
    print(f"→ {outdir}/per_gene_pcc{suff}.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="path2space_lopo_833")
    ap.add_argument("--smooth", action="store_true", help="Path2Space 空间平滑变体")
    a = ap.parse_args()
    score(tag=a.tag, smooth=a.smooth)
