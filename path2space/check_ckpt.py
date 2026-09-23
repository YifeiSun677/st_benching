"""核对 checkpoint：(1) 每 fold 的 ik 文件齐全、可 strict 载入；
(2) 用 checkpoint 重建集成、重新预测，与保存的 preds 对比（证明存下的就是被打分的权重）。

  python -m path2space.check_ckpt --tag path2space_lopo_833_ckpt              # 全部 8 fold
  python -m path2space.check_ckpt --tag path2space_lopo_833_ckpt --patients B E
"""
import argparse
import numpy as np
import torch
from .config import OUT_DIR, CKPT_DIR
from .dataset import load_section
from .train_mlp import predict
from .run_lopo import sections_by_patient, load_ik

def main(tag, patients=None, exp_ik=7, exp_il=7, exp_epochs=200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbp = sections_by_patient()
    patients = patients or sorted(sbp)
    ok_all = True
    for P in patients:
        fold = CKPT_DIR / tag / f"fold_{P}"
        files = sorted(fold.glob("ik_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
        n_ik = len(files)
        if n_ik == 0:
            print(f"fold {P}: NO checkpoint files in {fold} | FAIL"); ok_all = False; continue
        acc = {}
        for f in files:
            models, meta = load_ik(f, device)
            for s in sbp[P]:
                feat = load_section(s)["feat"].astype(np.float32)
                m_pred = np.mean([predict(m, feat, device) for m in models], axis=0)
                acc[s] = acc.get(s, 0) + m_pred
        worst = 0.0
        for s in sbp[P]:
            saved = np.load(OUT_DIR.parent / tag / "preds" / f"{s}.npz")["pred"]
            worst = max(worst, float(np.abs(acc[s] / n_ik - saved).max()))
        ok = (n_ik == exp_ik and len(meta["state_dicts"]) == exp_il
              and meta["epochs"] == exp_epochs and worst < 1e-4)
        ok_all &= ok
        print(f"fold {P}: ik files={n_ik} | MLPs/ik={len(meta['state_dicts'])} | "
              f"epochs={meta['epochs']} | max|re-pred − saved|={worst:.2e} | {'OK' if ok else 'FAIL'}")
    print("ALL OK" if ok_all else "SOMETHING FAILED")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--n_ik", type=int, default=7)       # 期望值；smoke test 时改小
    ap.add_argument("--n_il", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    main(a.tag, a.patients, a.n_ik, a.n_il, a.epochs)
