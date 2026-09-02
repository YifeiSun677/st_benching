"""per-fold per-section per-gene PCC + median SSE ratio + frac_beat_baseline + pred SD / true SD.

The npz layout matches the histogene port, so your existing scoring code can read
these files unchanged.
"""
import argparse, json
import numpy as np, pandas as pd
from pathlib import Path
from . import config as C


def score_section(f):
    d = np.load(f, allow_pickle=True)
    p, t = d["pred"].astype(np.float64), d["truth"].astype(np.float64)
    base = t.mean(0, keepdims=True)                      # per-gene mean = L2-optimal constant
    ps, ts = p.std(0), t.std(0)
    ok = (ps > 1e-12) & (ts > 1e-12)
    pcc = np.full(p.shape[1], np.nan)
    pc = ((p - p.mean(0)) * (t - t.mean(0))).mean(0)
    pcc[ok] = (pc / (ps * ts))[ok]
    sse = ((p - t) ** 2).sum(0)
    sse0 = ((base - t) ** 2).sum(0)
    ratio = np.where(sse0 > 0, sse / np.maximum(sse0, 1e-12), np.nan)
    sdr = np.where(ts > 1e-12, ps / np.maximum(ts, 1e-12), np.nan)
    return dict(section=f.stem, n_spots=p.shape[0],
                pcc=np.nanmean(pcc), pcc_median=np.nanmedian(pcc),
                frac_pos=float(np.nanmean(pcc > 0)),
                sse_ratio_median=float(np.nanmedian(ratio)),
                frac_beat_baseline=float(np.nanmean(ratio < 1)),
                sd_ratio_median=float(np.nanmedian(sdr))), pcc, d["genes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hist2st_lopo_833")
    args = ap.parse_args()
    root = C.OUT_ROOT / args.tag

    rows, per_gene, genes = [], [], None
    for fd in sorted(root.glob("fold*")):
        for f in sorted((fd / "preds").glob("*.npz")):
            r, pcc, genes = score_section(f)
            r["fold"] = fd.name
            rows.append(r); per_gene.append(pcc)
    df = pd.DataFrame(rows)
    df["patient"] = df.section.str[0]
    df.to_csv(root / "per_section_summary.csv", index=False)

    g = pd.DataFrame(np.vstack(per_gene).T, index=genes,
                     columns=[r["section"] for r in rows])
    g.mean(1).to_csv(root / "per_gene_pcc.csv", header=["pcc"])

    print(df.groupby("patient")[["pcc", "sse_ratio_median",
                                 "frac_beat_baseline", "sd_ratio_median"]].mean().round(4))
    head = dict(pcc_mean=float(df.pcc.mean()), pcc_median=float(df.pcc_median.median()),
                frac_pos=float(df.frac_pos.mean()),
                sse_ratio_median=float(df.sse_ratio_median.median()),
                frac_beat_baseline=float(df.frac_beat_baseline.mean()))
    for m in ["ERBB2", "GRB7", "ESR1", "PGR", "FASN", "GNAS", "MKI67"]:
        if m in g.index:
            head[m] = round(float(g.loc[m].mean()), 4)
    json.dump(head, open(root / "headline.json", "w"), indent=2)
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
