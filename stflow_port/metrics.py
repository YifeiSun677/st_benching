"""
stflow_port/metrics.py -- the benchmark's scoring footing, shared by train.py (probe
curve) and score.py.

  per-gene PCC, per fold: concatenate the held-out patient's sections, correlate each
      gene across spots (the "per-fold per-gene PCC averaged across folds" footing that
      replaced pooled-across-folds PCC). Genes with zero variance in truth or prediction
      are undefined (NaN) and excluded from means/medians; their count is reported.
  SSE ratio, per section: baseline = that gene's mean IN THAT SECTION (L2-optimal
      constant, so a constant predictor can never go below 1). Per gene we sum SSE and
      SST over the patient's sections, then report the median across genes and the
      fraction of genes with ratio < 1 (frac_genes_beat_baseline).
  sd ratio: per (section, gene) sd(pred)/sd(truth), median over valid pairs.
"""
import warnings

import numpy as np


def pcc_per_gene(pred, truth):
    p = pred - pred.mean(0, keepdims=True)
    t = truth - truth.mean(0, keepdims=True)
    num = (p * t).sum(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den <= 1e-12] = np.nan
    return r


def patient_metrics(preds, truths, genes, markers=()):
    """preds/truths: lists of [N_s, G] arrays, one per section of ONE held-out patient.
    (A constant predictor - e.g. the train-mean control - has undefined PCC: NaN, silently.)"""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return _patient_metrics(preds, truths, genes, markers)


def _patient_metrics(preds, truths, genes, markers):
    P = np.concatenate(preds, 0).astype(np.float64)
    T = np.concatenate(truths, 0).astype(np.float64)
    r = pcc_per_gene(P, T)
    ok = ~np.isnan(r)
    sse = np.zeros(P.shape[1])
    sst = np.zeros(P.shape[1])
    sd_ratios = []
    sec_pcc = []
    for p, t in zip(preds, truths):
        p = p.astype(np.float64); t = t.astype(np.float64)
        sse += ((p - t) ** 2).sum(0)
        sst += ((t - t.mean(0, keepdims=True)) ** 2).sum(0)
        sdt, sdp = t.std(0), p.std(0)
        v = sdt > 1e-12
        sd_ratios.append(sdp[v] / sdt[v])
        sec_pcc.append(np.nanmean(pcc_per_gene(p, t)))
    valid = sst > 1e-12
    ratio = np.full(P.shape[1], np.nan)
    ratio[valid] = sse[valid] / sst[valid]
    out = dict(
        n_spots=int(P.shape[0]), n_sections=len(preds),
        pcc_mean=float(np.nanmean(r)), pcc_median=float(np.nanmedian(r)),
        frac_pos=float((r[ok] > 0).mean()), n_pcc_undefined=int((~ok).sum()),
        pcc_section_mean=float(np.nanmean(sec_pcc)),
        sse_ratio_median=float(np.nanmedian(ratio)),
        frac_genes_beat_baseline=float((ratio[valid] < 1).mean()),
        sd_ratio_median=float(np.median(np.concatenate(sd_ratios))) if sd_ratios else float("nan"),
    )
    gi = {g: i for i, g in enumerate(genes)}
    for m in markers:
        if m in gi:
            out[f"pcc_{m}"] = float(r[gi[m]])
    return out, r, ratio
