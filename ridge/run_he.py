"""
ridge/run_he.py -- the her2st ridge baseline (UNI, local) on He et al. 2020 (subtype transfer arm).

No GPU: the features are the UNI 112-um embeddings STFlow's He arm already cached
(external/run_stflow_he.py -> /workspace/ext/he/features/stflow/<SEC>.npz, keys feat, spot_id), i.e. the
same crop / resize / UNI the her2st uni_hest112 features came from.  Each of the 8 her2st folds
(ridge.external_fit: refitted on its 7 training patients with its stored alpha) predicts every He section.

  truth   log1p(panel CP10K) of the He counts -- the same 'panel_cp10k_log1p' truth STFlow's He arm stores,
          so the existing scorer puts ridge on the same footing as the deep models
  pred    ridge on target cp10k_panel_ln (same scale)

writes  <out>/fold0<k>_<P>/preds/<SEC>.npz      (external/he_common.write_preds, model 'ridge_uni', cohort 'he')
then    python external/score_he.py --models ridge_uni      (as for every he_<model> run)

usage   cd /workspace/st_benching && python -m ridge.run_he --verify [--sections BC23287_C1] [--out ...]
        missing features -> GPU once: python external/run_stflow_he.py --features_only
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from . import external_fit as EF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external"))
import common as K  # noqa: E402
import he_common as H  # noqa: E402

FEAT_HE = H.HE / "features" / "stflow"


def panel_cp10k_ln(cnt, panel):
    C = cnt.reindex(columns=panel, fill_value=0).to_numpy(np.float64)
    s = C.sum(1, keepdims=True)
    return np.log1p(np.divide(C, s, out=np.zeros_like(C), where=s > 0) * 1e4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="*", default=None, help="default: every exported He section")
    ap.add_argument("--out", default=str(K.WS / "runs" / "he_ridge_uni"))
    ap.add_argument("--verify", action="store_true", help="first check the refitted folds on her2st")
    a = ap.parse_args()

    folds, meta, cache = EF.fit_folds()
    if a.verify and not EF.verify(folds, meta, cache):
        raise SystemExit("verify FAILED: the refitted ridge does not reproduce the her2st run")
    panel = meta["genes"]
    assert panel == K.load_panel(), "ridge panel order != external panel order"

    exported = {f.name[len("counts_"):-len(".npz")] for f in H.HE_CALIB.glob("counts_*.npz")}
    secs = a.sections or [s for s in H.he_sections() if s in exported]
    absent = [s for s in secs if not (FEAT_HE / f"{s}.npz").exists()]
    if absent:
        raise SystemExit(f"{len(absent)} He sections have no cached UNI features (e.g. {absent[:3]}). "
                         f"Run once on the GPU: python external/run_stflow_he.py --features_only")

    # her2st held-out sections into the same dir (cohort 'her2st'), for score_he.py's paired delta
    index, Xh, Yh = cache
    for P, f in folds.items():
        for sec in sorted(set(index.section[index.patient == P])):
            rows = index[index.section == sec]
            r = rows.row.values
            K.write_preds(os.path.join(a.out, f"fold0{f['fold']}_{P}"), sec, pred=EF.predict(f, Xh[r]),
                          truth=Yh[r], spot_ids=list(rows.spot_id.astype(str)), genes=panel,
                          trainmean=f["trainmean"], model="ridge_uni", fold=f["fold"])
    print(f"her2st held-out sections written for the paired delta ({index.section.nunique()} sections)")

    t0 = time.time()
    for sec in secs:
        z = np.load(FEAT_HE / f"{sec}.npz", allow_pickle=True)
        cnt = K.read_counts(sec, H.HE_DATA)
        sid = [str(s) for s in z["spot_id"]]
        assert sid == list(cnt.index.astype(str)), f"{sec}: cached feature rows != counts rows"
        truth = panel_cp10k_ln(cnt, panel).astype(np.float32)
        X = z["feat"].astype(np.float64)
        assert X.shape[1] == meta["dim"], f"{sec}: feature dim {X.shape[1]} != her2st {meta['dim']}"
        pcc = []
        for P, f in folds.items():
            pred = EF.predict(f, X).astype(np.float32)
            H.write_preds(os.path.join(a.out, f"fold0{f['fold']}_{P}"), sec, pred=pred, truth=truth,
                          spot_ids=sid, genes=panel, trainmean=f["trainmean"], model="ridge_uni",
                          fold=f["fold"],
                          extra=dict(meta={k: v for k, v in meta.items() if k != "genes"}, alpha=f["alpha"],
                                     features=str(FEAT_HE), truth="log1p(panel CP10K)", cohort="he"))
            pcc.append(np.nanmean(K.per_gene_pcc(pred, truth)))
        print(f"{sec}: {len(sid)} spots, mean PCC over folds {np.mean(pcc):.4f}", flush=True)
    print(f"DONE {len(secs)} He sections x {len(folds)} folds in {time.time() - t0:.0f}s -> {a.out}\n"
          f"score: python external/score_he.py --models ridge_uni")


if __name__ == "__main__":
    main()
