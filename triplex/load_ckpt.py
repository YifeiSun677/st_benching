"""
Load a saved TRIPLEX fold checkpoint for re-inference / perturbation runs.

  from triplex.load_ckpt import load_fold
  model, meta = load_fold("triplex_lopo_833_ckpt", "B")      # final.pt, eval mode
  model, meta = load_fold("triplex_lopo_833_ckpt", "B", which="epoch_100.pt")

  python -m triplex.load_ckpt --tag triplex_lopo_833_ckpt --patient B   # sanity check
"""
import os
import argparse
import torch

from . import config
from .triplex_import import build_model


def load_fold(tag, patient, which="final.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(config.CKPT_DIR, tag, f"fold_{patient}", which)
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, ck.get("meta", {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--patient", required=True)
    ap.add_argument("--which", default="final.pt")
    a = ap.parse_args()
    m, meta = load_fold(a.tag, a.patient, a.which)
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded {a.which} for fold {a.patient}: {n/1e6:.1f}M params, "
          f"epochs={meta.get('epochs')}, test={meta.get('test_sections')}")
