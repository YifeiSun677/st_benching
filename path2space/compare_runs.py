"""新旧两次 run 对比：输入一致？预测一致？headline 一致？

  python -m path2space.compare_runs --old path2space_lopo_833 --new path2space_lopo_833_ckpt
"""
import argparse
import numpy as np
from scipy.stats import spearmanr
from .config import OUT_DIR, MARKERS
from .score import per_gene_pcc

def load(tag):
    d = OUT_DIR.parent / tag / "preds"
    return {f.stem: np.load(f, allow_pickle=True) for f in sorted(d.glob("*.npz"))}

def summarise(runs):
    G = np.vstack([per_gene_pcc(z["pred"], z["truth"]) for z in runs.values()])
    genes = list(next(iter(runs.values()))["genes"])
    gm = np.nanmean(G, 0)
    pp = {}
    for (s, z), row in zip(runs.items(), G):
        pp.setdefault(str(z["patient"]), []).append(row)
    pp = {p: float(np.nanmean(np.vstack(v))) for p, v in sorted(pp.items())}
    mk = {m: float(np.nanmean(G[:, genes.index(m)])) for m in MARKERS if m in genes}
    return gm, pp, mk

def main(old, new):
    A, B = load(old), load(new)
    print(f"sections: old={len(A)} new={len(B)}")
    assert A.keys() == B.keys(), "section 集不同"
    same_in = all(np.array_equal(A[s]["truth"], B[s]["truth"]) and
                  np.array_equal(A[s]["spot_id"], B[s]["spot_id"]) and
                  np.array_equal(A[s]["genes"], B[s]["genes"]) for s in A)
    print(f"[1] inputs identical (truth, spot_id, genes): {'YES' if same_in else 'NO — stop here'}")
    dmax = max(float(np.abs(A[s]["pred"] - B[s]["pred"]).max()) for s in A)
    r = np.corrcoef(np.concatenate([A[s]["pred"].ravel() for s in A]),
                    np.concatenate([B[s]["pred"].ravel() for s in A]))[0, 1]
    print(f"[2] predictions: max|Δ|={dmax:.2e}  corr={r:.6f}")
    ga, pa, ma = summarise(A); gb, pb, mb = summarise(B)
    print(f"[3] headline mean  {np.nanmean(ga):+.4f} → {np.nanmean(gb):+.4f}")
    print(f"    headline median {np.nanmedian(ga):+.4f} → {np.nanmedian(gb):+.4f}")
    print(f"    frac_pos       {(ga>0).mean():.3f} → {(gb>0).mean():.3f}")
    print("    per-patient:  " + "  ".join(f"{p} {pa[p]:+.3f}→{pb[p]:+.3f}" for p in pa))
    print(f"    patient-order Spearman: {spearmanr(list(pa.values()), list(pb.values()))[0]:.3f}")
    print("    markers:      " + "  ".join(f"{m} {ma[m]:+.3f}→{mb[m]:+.3f}" for m in ma))
    m = ~np.isnan(ga) & ~np.isnan(gb)
    print(f"    per-gene PCC corr between runs: {np.corrcoef(ga[m], gb[m])[0,1]:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    a = ap.parse_args()
    main(a.old, a.new)
