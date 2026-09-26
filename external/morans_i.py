#!/usr/bin/env python
"""Stage 6.4 -- Moran's I of the ground truth, her2st and Visium, through ONE code path.

Per section: log1p(CP10K) over all genes -> top-150 most variable genes ->
Moran's I with inverse-distance weights (w_ij = 1/d_ij, i != j, unstandardised),
distances in her2st units (200 um): her2st uses new_x/new_y, Visium x_eq/y_eq.
Visium is also scored on 7-spot pseudo-spots (summed counts, centre position),
because the denser Visium grid alone changes Moran's I.

frac_sig = share of the 150 genes with z > 1.645 under the normality assumption
(Cliff & Ord).  This definition may differ from the earlier her2st analysis, so
her2st is RECOMPUTED here rather than reusing old numbers.

writes /workspace/results/ext/morans_i.csv  (section, cohort, patient, footing, n_spots,
       moran_median, moran_mean, frac_sig)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common as K

TOP = 150


def norm_log(X):
    lib = X.sum(1, keepdims=True)
    lib[lib == 0] = 1
    return np.log1p(X / lib * 1e4)


def morans(Y, xy):
    n = len(Y)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    with np.errstate(divide="ignore"):
        W = np.where(d > 0, 1.0 / d, 0.0)
    S0 = W.sum()
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(0) + W.sum(1)) ** 2).sum()
    Z = Y - Y.mean(0)
    num = np.einsum("ig,ij,jg->g", Z, W, Z, optimize=True)
    den = (Z ** 2).sum(0)
    I = (n / S0) * num / den
    EI = -1.0 / (n - 1)
    VI = (n * n * S1 - n * S2 + 3 * S0 ** 2) / ((n * n - 1) * S0 ** 2) - EI ** 2
    return I, (I - EI) / np.sqrt(VI)


def top_var(Xn, k=TOP):
    v = Xn.var(0)
    return np.argsort(v)[::-1][:k]


def one(sec, X, xy, cohort, footing):
    Xn = norm_log(X.astype(np.float64))
    Xn = Xn[:, top_var(Xn)]
    I, z = morans(Xn, xy.astype(np.float64))
    return dict(section=sec, cohort=cohort, patient=K.PATIENT_OF.get(sec, sec[0]), footing=footing,
                n_spots=len(X), moran_median=np.median(I), moran_mean=I.mean(), frac_sig=np.mean(z > 1.645))


def main():
    rows = []
    for sec in K.her2st_sections():
        cnt = K.read_counts(sec, K.HER2ST_ROOT)
        sp = K.read_spots(sec, K.HER2ST_ROOT)
        ids = [s for s in cnt.index if s in sp.index]
        xy = sp.loc[ids, ["new_x" if "new_x" in sp else "x", "new_y" if "new_y" in sp else "y"]].values
        rows.append(one(sec, cnt.loc[ids].values, xy, "her2st", "native"))
        print(rows[-1])
    for sec in K.VISIUM_SECTIONS:
        z = np.load(K.CALIB / f"counts_{sec}.npz", allow_pickle=True)
        X = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])).toarray()
        ids = list(z["spot_id"])
        sp = K.read_spots(sec, K.VIS_ROOT).loc[ids]
        xy = sp[["x_eq", "y_eq"]].values
        rows.append(one(sec, X, xy, "visium", "native"))
        print(rows[-1])
        agg, centres = K.aggregate_counts(X, ids, sec)
        rows.append(one(sec, agg, sp.loc[centres, ["x_eq", "y_eq"]].values, "visium", "pseudo7"))
        print(rows[-1])
    out = K.WS / "results" / "ext"
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "morans_i.csv", index=False)
    print(df.groupby(["cohort", "patient", "footing"])[["moran_median", "frac_sig"]].median().round(4))


if __name__ == "__main__":
    main()
