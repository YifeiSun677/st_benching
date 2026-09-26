#!/usr/bin/env python
"""Stage 5 -- template every model driver follows (run_bleep.py is the worked example).

Per fold P (held-out her2st patient, fold index k):
  1. load the fold's main-table weights                        -> load_model(P)
  2. trainmean = mean of the model's OWN truth over the fold's 7 training patients
  3. for sec in P's her2st sections (round-trip) + I1 I2 J1 K1:
       pred, truth, spot_ids = predict(model, sec, root)       (root = VIS_ROOT for Visium)
       Visium only: truth_ps = transform(sum of raw counts over each 7-spot group)
       K.write_preds(out/fold0k_P, sec, ...)
  4. her2st sections: compare pred/PCC with the stored LOPO output -> ext/roundtrip.tsv

Rules every driver must keep:
  * Positions: Visium spot files carry x,y = Visium ARRAY indices (unique ids only).
    Any model that uses positions must read new_x/new_y (= x_eq/y_eq, 200-um units) or
    x_int/y_int (integer embeddings) instead of x,y.
  * Never list sections by globbing a data root that mixes cohorts; folds come from
    common.lopo_folds() (her2st only).
  * Truth comes from the port's own transform code, never re-implemented here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402


def load_model(patient):            # -> model in eval mode
    raise NotImplementedError


def train_truth_mean(fold):         # -> (833,) mean of own-space truth over fold['train']
    raise NotImplementedError


def predict(model, sec, root):      # -> pred (n x 833), truth (n x 833), spot_ids (n)
    raise NotImplementedError


def transform(raw_counts_panel):    # (m x 833 raw) -> own target space, e.g. cpm_log1p
    raise NotImplementedError
