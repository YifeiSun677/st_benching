"""LOPO（8 fold）× (ik×il) 集成。每 fold：训练集成 → 预测 held-out patient 全部 section。"""
import argparse, json
from collections import defaultdict
import numpy as np
import torch
from .config import (FEATURE_DIR, OUT_DIR, N_IK, N_IL, SEED)
from .dataset import build_xy, load_section, transform_counts
from .train_mlp import train_one_mlp, predict
from .p2s_import import init_random_seed

def sections_by_patient():
    man = json.loads((FEATURE_DIR / "manifest.json").read_text())
    d = defaultdict(list)
    for sec, info in man.items():
        d[info["patient"]].append(sec)
    return {p: sorted(v) for p, v in d.items()}

def make_ik_subsets(train_patients, n_ik):
    """ik 子集：n_ik<=1 → 用全部训练 patient；否则每个子集丢掉一个 patient
    （=leave-one-train-patient-out）。"""
    if n_ik is None:
        n_ik = len(train_patients)     # 默认：对训练 patient 做 LOPO-within-train
    if n_ik <= 1:
        return [list(train_patients)]
    subsets = []
    for i in range(n_ik):
        drop = train_patients[i % len(train_patients)]
        subsets.append([p for p in train_patients if p != drop])
    return subsets

def run(n_ik=N_IK, n_il=N_IL, epochs=None, tag="path2space_lopo_833", only_patients=None):
    init_random_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbp = sections_by_patient()
    patients = sorted(sbp)
    if only_patients:
        patients = [p for p in patients if p in only_patients]
    out = OUT_DIR.parent / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "preds").mkdir(exist_ok=True)

    for held in patients:                      # ---- 外层 LOPO ----
        train_pat = [p for p in patients if p != held]
        test_secs = sbp[held]
        ik_subsets = make_ik_subsets(train_pat, n_ik)
        print(f"\n=== fold held={held} | train patients={train_pat} | "
              f"ik={len(ik_subsets)} × il={n_il} = {len(ik_subsets)*n_il} MLPs ===")

        # 预取每个 test section 的特征
        test_feat = {s: load_section(s)["feat"].astype(np.float32) for s in test_secs}
        n_genes = load_section(test_secs[0])["counts833"].shape[1]
        acc = {s: np.zeros((test_feat[s].shape[0], n_genes), np.float64) for s in test_secs}

        for ik, subset in enumerate(ik_subsets):        # ---- ik ----
            secs = sum((sbp[p] for p in subset), [])
            X, Y = build_xy(secs)
            preds_il = {s: [] for s in test_secs}
            for il in range(n_il):                       # ---- il ----
                model = train_one_mlp(X, Y, seed=SEED + 1000*ik + il,
                                      device=device, epochs=epochs)
                for s in test_secs:
                    preds_il[s].append(predict(model, test_feat[s], device))
            for s in test_secs:                          # 内层 il 均值
                acc[s] += np.mean(preds_il[s], axis=0)
            print(f"  ik={ik} (train={subset}) done")

        for s in test_secs:                              # 外层 ik 均值
            pred = acc[s] / len(ik_subsets)
            d = load_section(s)
            np.savez_compressed(
                out / "preds" / f"{s}.npz",
                pred=pred.astype(np.float32),
                truth=transform_counts(d["counts833"].astype(np.float32)).astype(np.float32),
                counts_raw=d["counts833"].astype(np.float32),
                spot_id=d["spot_id"], ax=d["ax"], ay=d["ay"], genes=d["genes"],
                patient=held, n_ik=len(ik_subsets), n_il=n_il,
            )
        print(f"  fold {held} 完成，写出 {len(test_secs)} 个 section 预测")

    (out / "config_used.json").write_text(json.dumps(
        {"n_ik": n_ik, "n_il": n_il, "epochs": epochs}, indent=2))
    print(f"\nLOPO 完成 → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ik", type=int, default=N_IK)
    ap.add_argument("--n_il", type=int, default=N_IL)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default="path2space_lopo_833")
    ap.add_argument("--patients", nargs="*", default=None)
    a = ap.parse_args()
    run(n_ik=a.n_ik, n_il=a.n_il, epochs=a.epochs, tag=a.tag, only_patients=a.patients)
