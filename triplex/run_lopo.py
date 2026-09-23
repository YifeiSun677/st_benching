"""
Run the full 8-fold LOPO sweep, then score.

  python -m triplex.run_lopo --tag triplex_lopo_833
  python -m triplex.run_lopo --tag triplex_smoke --patients B --epochs 3   # smoke
  python -m triplex.run_lopo --tag triplex_lopo_833_ckpt --resume          # after a pod dies

With --resume, a fold whose final.pt AND preds/ already exist is skipped, and a
fold with a resume.pt continues from its last saved epoch.
"""
import os
import glob
import argparse
from . import config
from .train import train_fold
from . import score


def _fold_done(tag, patient):
    ck = os.path.join(config.CKPT_DIR, tag, f"fold_{patient}", "final.pt")
    preds = glob.glob(os.path.join(config.OUTPUT_DIR, tag, f"fold_{patient}", "preds", "*.npz"))
    return os.path.isfile(ck) and len(preds) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="triplex_lopo_833")
    ap.add_argument("--patients", nargs="*", default=config.PATIENTS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_score", action="store_true")
    a = ap.parse_args()

    for p in a.patients:
        if a.resume and _fold_done(a.tag, p):
            print(f"[run_lopo] fold {p} already complete -> skip")
            continue
        train_fold(p, a.tag, epochs=a.epochs, resume=a.resume)

    if not a.no_score:
        score.score_run(a.tag)


if __name__ == "__main__":
    main()
