"""
Run every fold of a protocol back to back.

  python run_lopo.py --protocol lopo --epochs 50            # full 8-fold LOPO
  python run_lopo.py --protocol loso --epochs 50            # 32-fold reproduction
  python run_lopo.py --protocol lopo --epochs 50 --only B   # single patient

Each fold's outputs land in OUT_DIR/<tag>/<fold>/. Safe to resume: folds whose
run.json already exists are skipped unless --force.
"""
import os
import argparse

import config as C
import her2_data as H
from train_tcgn import train_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["lopo", "loso"], default="lopo")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these fold names")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    folds = H.lopo_folds() if args.protocol == "lopo" else H.loso_folds()
    tag = args.tag or ("tcgn_%s_833_e%d" % (args.protocol, args.epochs))
    for name, te, tr in folds:
        if args.only and name not in args.only:
            continue
        done = os.path.exists(os.path.join(C.OUT_DIR, tag, name, "run.json"))
        if done and not args.force:
            print("[skip] %s already has run.json" % name)
            continue
        print("\n==================== fold %s ====================" % name)
        train_fold(name, te, tr, tag, epochs=args.epochs)


if __name__ == "__main__":
    main()
