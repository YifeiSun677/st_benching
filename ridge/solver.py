"""
ridge/solver.py -- closed-form ridge from per-patient sufficient statistics,
plus the benchmark's PCC definition.

Why sufficient statistics: nested leave-one-patient-out needs 8 outer x 7 inner
fits. Each fit only needs sums over its training patients (n, sum x, sum y, X'X,
X'Y), so per-patient sums are computed once and every fit is a d x d eigen-
decomposition. Standardisation (z-score of X, centring of Y) is still computed
from each fit's OWN training patients, exactly as if X were re-scaled per fit.

For one fit with standardised Xs (n x d) and centred Yc (n x G):
    Xs'Xs = V diag(lam) V'          (eigendecomposition)
    W(alpha) = V diag(1 / (lam + alpha)) V' Xs'Yc
so every alpha on the grid costs one diagonal rescale.
"""
from __future__ import annotations

import warnings

import numpy as np


# ------------------------------------------------------------------ metric
def gene_pcc(pred, truth) -> np.ndarray:
    """Per-gene Pearson r over the spots of ONE section. NaN where either side has
    zero variance -- identical to the benchmark scorer's _pcc."""
    p = np.asarray(pred, np.float64)
    t = np.asarray(truth, np.float64)
    p = p - p.mean(0)
    t = t - t.mean(0)
    num = (p * t).sum(0)
    den = np.sqrt((p * p).sum(0) * (t * t).sum(0))
    out = np.full(p.shape[1], np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def sections_pcc(pred, truth, sections) -> np.ndarray:
    """Per-gene PCC averaged (nanmean) over sections -> the fold-level vector."""
    per = [gene_pcc(pred[sections == s], truth[sections == s]) for s in np.unique(sections)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN gene -> NaN, silently
        return np.nanmean(per, 0)


# ------------------------------------------------------------------ sufficient stats
class Stats:
    """Sums over one patient's spots (float64)."""
    __slots__ = ("n", "sx", "sy", "xx", "xy")

    def __init__(self, X, Y):
        X = np.asarray(X, np.float64)
        Y = np.asarray(Y, np.float64)
        self.n = len(X)
        self.sx = X.sum(0)
        self.sy = Y.sum(0)
        self.xx = X.T @ X
        self.xy = X.T @ Y

    @staticmethod
    def total(parts):
        t = Stats.__new__(Stats)
        t.n = sum(p.n for p in parts)
        t.sx = sum(p.sx for p in parts)
        t.sy = sum(p.sy for p in parts)
        t.xx = sum(p.xx for p in parts)
        t.xy = sum(p.xy for p in parts)
        return t


class RidgeFit:
    """Ridge fitted on the union of some patients' Stats, for a whole alpha grid."""

    def __init__(self, stats: Stats):
        n = stats.n
        self.mx = stats.sx / n
        self.my = stats.sy / n
        cxx = stats.xx - n * np.outer(self.mx, self.mx)          # centred scatter
        cxy = stats.xy - n * np.outer(self.mx, self.my)
        var = np.clip(np.diag(cxx) / n, 0, None)
        self.sd = np.where(var > 1e-12, np.sqrt(var), 1.0)      # constant feature -> left as 0
        sxx = cxx / np.outer(self.sd, self.sd)
        sxy = cxy / self.sd[:, None]
        lam, V = np.linalg.eigh((sxx + sxx.T) / 2)
        self.lam = np.clip(lam, 0, None)
        self.V = V
        self.A = V.T @ sxy                                       # d x G

    def project(self, X):
        """Standardise with THIS fit's training statistics, rotate to the eigenbasis."""
        return ((np.asarray(X, np.float64) - self.mx) / self.sd) @ self.V

    def predict(self, X, alpha, B=None):
        B = self.project(X) if B is None else B
        return B @ (self.A / (self.lam + alpha)[:, None]) + self.my

    def coef(self, alpha):
        """Weights on the ORIGINAL feature scale (d x G) and intercept (G,)."""
        W_std = self.V @ (self.A / (self.lam + alpha)[:, None])
        W = W_std / self.sd[:, None]
        return W, self.my - self.mx @ W


# ------------------------------------------------------------------ nested selection
def choose_alpha(X, Y, patient, sections, train_patients, alphas):
    """Inner leave-one-patient-out over `train_patients` only.
    Score = mean over inner folds of (mean over genes of the section-averaged PCC),
    the same aggregation as the headline. Ties go to the LARGER alpha.
    Returns (best_alpha, scores [n_alpha], per_inner {patient: [n_alpha]})."""
    stats = {p: Stats(X[patient == p], Y[patient == p]) for p in train_patients}
    per_inner = {}
    for q in train_patients:
        fit = RidgeFit(Stats.total([stats[p] for p in train_patients if p != q]))
        m = patient == q
        B = fit.project(X[m])
        row = []
        for a in alphas:
            pcc = sections_pcc(fit.predict(None, a, B=B), Y[m], sections[m])
            row.append(np.nanmean(pcc))
        per_inner[q] = np.array(row)
    scores = np.mean(np.vstack(list(per_inner.values())), 0)
    safe = np.where(np.isfinite(scores), np.round(scores, 10), -np.inf)
    best = max(range(len(alphas)), key=lambda i: (safe[i], alphas[i]))
    return alphas[best], scores, per_inner, stats
