"""LOPO（8 fold）× (ik×il) 集成。每 fold：训练集成 → 预测 held-out patient 全部 section。

  python -m path2space.run_lopo --tag path2space_lopo_833_ckpt              # 全量，存 checkpoint
  python -m path2space.run_lopo --tag path2space_lopo_833_ckpt --resume     # pod 挂了之后原样再跑
  python -m path2space.run_lopo --tag p2s_ckpt_smoke --patients A --n_ik 1 --n_il 1 --epochs 2

checkpoint：CKPT_DIR/<tag>/fold_<P>/ik_<k>.pt（该 ik 的 N_IL 个 MLP 权重 + seeds/子集/epochs）。
--resume：已有 ik_<k>.pt 的 ik 直接载入权重做预测，不重训；preds 已全在的 fold 整个跳过。
保存不消耗 RNG，且每个 MLP 训练前单独 seed，所以有无 checkpoint / 是否续跑，训练完全一致。
"""
import argparse, json, os, time
from collections import defaultdict
import numpy as np
import torch
from .config import (FEATURE_DIR, OUT_DIR, N_IK, N_IL, SEED, EPOCHS, DROPOUT, CKPT_DIR)
from .dataset import build_xy, load_section, transform_counts
from .train_mlp import train_one_mlp, predict
from .p2s_import import init_random_seed, MLP_regression_relu_two

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

# ---------------------------------------------------------------- checkpoints
def _atomic_save(obj, path):
    """先写 .tmp 再 rename：pod 在写盘途中挂掉也不会留下截断的 checkpoint。"""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def build_mlp(n_in, n_out, device):
    """与训练时同构的空模型（bias_init 只是初始化，load_state_dict 会覆盖）。"""
    return MLP_regression_relu_two(n_inputs=n_in, n_hiddens=n_in, n_outputs=n_out,
                                   dropout=DROPOUT, bias_init=torch.zeros(n_out)).to(device)

def load_ik(path, device):
    """读 ik_<k>.pt → (模型列表[eval], meta)。strict=True：权重形状/键必须完全对上。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    models = []
    for sd in ck["state_dicts"]:
        m = build_mlp(ck["n_in"], ck["n_out"], device)
        m.load_state_dict(sd, strict=True)
        m.eval()
        models.append(m)
    return models, ck

# ---------------------------------------------------------------- main
def run(n_ik=N_IK, n_il=N_IL, epochs=None, tag="path2space_lopo_833", only_patients=None,
        resume=False, ckpt_dir=CKPT_DIR):
    init_random_seed(SEED)
    epochs = epochs or EPOCHS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbp = sections_by_patient()
    patients = sorted(sbp)
    all_patients = list(patients)
    if only_patients:
        patients = [p for p in patients if p in only_patients]
    out = OUT_DIR.parent / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "preds").mkdir(exist_ok=True)
    ck_root = ckpt_dir / tag
    ck_root.mkdir(parents=True, exist_ok=True)
    print(f"preds → {out}\nckpt  → {ck_root}\nepochs={epochs} n_il={n_il} resume={resume}")

    for held in patients:                      # ---- 外层 LOPO ----
        t0 = time.time()
        train_pat = [p for p in all_patients if p != held]
        test_secs = sbp[held]
        ik_subsets = make_ik_subsets(train_pat, n_ik)
        fold_ck = ck_root / f"fold_{held}"
        fold_ck.mkdir(exist_ok=True)

        if resume and all((out / "preds" / f"{s}.npz").exists() for s in test_secs) \
                and all((fold_ck / f"ik_{k}.pt").exists() for k in range(len(ik_subsets))):
            print(f"\n=== fold held={held}: preds + 全部 ik checkpoint 已在，跳过 ===")
            continue

        print(f"\n=== fold held={held} | train patients={train_pat} | "
              f"ik={len(ik_subsets)} × il={n_il} = {len(ik_subsets)*n_il} MLPs ===")

        test_feat = {s: load_section(s)["feat"].astype(np.float32) for s in test_secs}
        n_genes = load_section(test_secs[0])["counts833"].shape[1]
        acc = {s: np.zeros((test_feat[s].shape[0], n_genes), np.float64) for s in test_secs}

        for ik, subset in enumerate(ik_subsets):        # ---- ik ----
            ck_path = fold_ck / f"ik_{ik}.pt"
            seeds = [SEED + 1000*ik + il for il in range(n_il)]
            preds_il = {s: [] for s in test_secs}

            if resume and ck_path.exists():
                models, meta = load_ik(ck_path, device)
                assert meta["seeds"] == seeds and meta["train_patients"] == subset \
                    and meta["epochs"] == epochs, f"{ck_path} 与当前设置不符：{meta['seeds']}"
                for m in models:
                    for s in test_secs:
                        preds_il[s].append(predict(m, test_feat[s], device))
                print(f"  ik={ik} (train={subset}) 从 checkpoint 载入")
            else:
                secs = sum((sbp[p] for p in subset), [])
                X, Y = build_xy(secs)
                sds = []
                for seed in seeds:                       # ---- il ----
                    model = train_one_mlp(X, Y, seed=seed, device=device, epochs=epochs)
                    for s in test_secs:
                        preds_il[s].append(predict(model, test_feat[s], device))
                    sds.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                _atomic_save(dict(state_dicts=sds, seeds=seeds, train_patients=subset,
                                  held_out=held, ik=ik, epochs=epochs, n_in=X.shape[1],
                                  n_out=Y.shape[1], n_train_spots=int(X.shape[0])), ck_path)
                print(f"  ik={ik} (train={subset}) done → {ck_path.name}")

            for s in test_secs:                          # 内层 il 均值
                acc[s] += np.mean(preds_il[s], axis=0)

        for s in test_secs:                              # 外层 ik 均值
            pred = acc[s] / len(ik_subsets)
            d = load_section(s)
            dst = out / "preds" / f"{s}.npz"
            tmp = str(dst) + ".tmp"
            with open(tmp, "wb") as fh:                  # 原子写：pod 中途挂掉不会留下半个 npz
                np.savez_compressed(
                    fh,
                    pred=pred.astype(np.float32),
                    truth=transform_counts(d["counts833"].astype(np.float32)).astype(np.float32),
                    counts_raw=d["counts833"].astype(np.float32),
                    spot_id=d["spot_id"], ax=d["ax"], ay=d["ay"], genes=d["genes"],
                    patient=held, n_ik=len(ik_subsets), n_il=n_il,
                    )
            os.replace(tmp, dst)
        print(f"  fold {held} 完成，写出 {len(test_secs)} 个 section 预测 ({(time.time()-t0)/60:.1f} min)")

    (out / "config_used.json").write_text(json.dumps(
        {"n_ik": n_ik, "n_il": n_il, "epochs": epochs, "seed": SEED,
         "ckpt_dir": str(ck_root)}, indent=2))
    print(f"\nLOPO 完成 → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ik", type=int, default=N_IK)
    ap.add_argument("--n_il", type=int, default=N_IL)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default="path2space_lopo_833")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    run(n_ik=a.n_ik, n_il=a.n_il, epochs=a.epochs, tag=a.tag,
        only_patients=a.patients, resume=a.resume)
