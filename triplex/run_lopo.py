"""
Run the full 8-fold LOPO sweep, then score.

  python -m triplex.run_lopo --tag triplex_lopo_833
  python -m triplex.run_lopo --tag triplex_smoke --patients B --epochs 3   # smoke
"""
import argparse
from . import config
from .train import train_fold
from . import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="triplex_lopo_833")
    ap.add_argument("--patients", nargs="*", default=config.PATIENTS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no_score", action="store_true")
    a = ap.parse_args()

    for p in a.patients:
        train_fold(p, a.tag, epochs=a.epochs)

    if not a.no_score:
        score.score_run(a.tag)


if __name__ == "__main__":
    main()
