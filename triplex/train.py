"""
Train one LOPO fold: hold out one patient, train on the other seven, dump
per-section predictions for the held-out patient at the LAST epoch of a fixed,
pre-declared budget (no selection on held-out data -- benchmark-wide rule).

Fold outputs: OUTPUT_DIR/<tag>/fold_<patient>/
    run.json           config, timing, train/test section lists
    preds/<section>.npz  pred (N,833), truth (N,833), genes, coords, spot_id
"""
import os
import json
import time
import argparse
import numpy as np
import torch

from . import config, her2st
from .dataset import TriTrainDataset, TriTestSections
from .triplex_import import build_model


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_fold(patient, tag, epochs=None, lr=None, batch_size=None, device=None):
    epochs = epochs or config.EPOCHS
    lr = lr or config.LR
    batch_size = batch_size or config.BATCH_SIZE
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    panel = her2st.load_panel()
    index_df = her2st.load_spot_index()
    test_sections = her2st.sections_for_patient(index_df, patient)
    train_sections = [s for s in her2st.all_sections(index_df)
                      if s not in test_sections]

    print(f"[fold {patient}] train {len(train_sections)} sections, "
          f"test {len(test_sections)} sections")

    train_ds = TriTrainDataset(train_sections, panel)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=False, drop_last=False)

    model = build_model().to(device)
    if config.FREEZE_TARGET:
        for p in model.target_encoder.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=config.WEIGHT_DECAY)

    t0 = time.time()
    per_epoch = []
    for ep in range(epochs):
        model.train()
        te, running = time.time(), 0.0
        for batch in loader:
            batch = _to_device(batch, device)
            out = model(img=batch["img"], mask=batch["mask"],
                        neighbor_emb=batch["neighbor_emb"],
                        pid=batch["pid"], sid=batch["sid"],
                        label=batch["label"], dataset=train_ds, phase="train")
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss) * len(batch["pid"])
        avg = running / len(train_ds)
        dt = time.time() - te
        per_epoch.append(avg)
        if ep < 3 or (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"[fold {patient}] epoch {ep+1}/{epochs}  loss {avg:.4f}  {dt:.1f}s")

    # ---- last-epoch inference on the held-out patient ----
    out_dir = os.path.join(config.OUTPUT_DIR, tag, f"fold_{patient}")
    os.makedirs(os.path.join(out_dir, "preds"), exist_ok=True)
    test_ds = TriTestSections(test_sections, panel)
    model.eval()
    with torch.no_grad():
        for s in test_sections:
            b = test_ds.section_batch(s, device=device)
            res = model(img=b["img"], mask=b["mask"],
                        neighbor_emb=b["neighbor_emb"],
                        position=b["position"], global_emb=b["global_emb"])
            pred = res["logits"].cpu().numpy().astype(np.float32)
            np.savez(os.path.join(out_dir, "preds", f"{s}.npz"),
                     pred=pred, truth=b["label"].astype(np.float32),
                     genes=np.array(panel), coords=b["coords"],
                     spot_id=b["spot_id"], section=s)
            print(f"[fold {patient}] wrote preds/{s}.npz  {pred.shape}")

    run = dict(patient=patient, tag=tag, epochs=epochs, lr=lr,
               batch_size=batch_size, cpm=config.CPM, smooth=config.SMOOTH,
               freeze_target=config.FREEZE_TARGET, pos_layer=config.POS_LAYER,
               res_neighbor=list(config.RES_NEIGHBOR),
               train_sections=train_sections, test_sections=test_sections,
               train_loss_last=per_epoch[-1], seconds_total=time.time() - t0)
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2)
    print(f"[fold {patient}] done in {run['seconds_total']/60:.1f} min -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--tag", default="triplex_lopo_833")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    a = ap.parse_args()
    train_fold(a.patient, a.tag, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size)
