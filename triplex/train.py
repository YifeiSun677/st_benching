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


# ---------------------------------------------------------------- checkpoints
def _atomic_save(obj, path):
    """Write to a temp file then rename, so a pod dying mid-write never leaves a
    truncated checkpoint in place of a good one."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _save_resume(ckpt_dir, model, opt, epoch_done, per_epoch, probe_curve, meta):
    _atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                      epoch=epoch_done, per_epoch=per_epoch,
                      probe_curve=probe_curve, rng=_rng_state(), meta=meta),
                 os.path.join(ckpt_dir, "resume.pt"))


def _heldout_pcc(model, test_ds, test_sections, device):
    """Quick calibration metric: per-section per-gene Pearson r, averaged across
    the held-out sections then over genes (nan-safe). Same spirit as score.py;
    used only to pick a fixed epoch budget, NOT for final scoring."""
    was_training = model.training
    model.eval()
    per_sec = []
    with torch.no_grad():
        for s in test_sections:
            b = test_ds.section_batch(s, device=device)
            pred = model(img=b["img"], mask=b["mask"], neighbor_emb=b["neighbor_emb"],
                         position=b["position"], global_emb=b["global_emb"])["logits"]
            pred = pred.cpu().numpy().astype(np.float64)
            truth = np.asarray(b["label"], dtype=np.float64)
            pv = pred - pred.mean(0); tv = truth - truth.mean(0)
            denom = np.sqrt((pv**2).sum(0) * (tv**2).sum(0))
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (pv * tv).sum(0) / denom          # (833,), nan where zero var
            per_sec.append(r)
    if was_training:
        model.train()
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(np.nanmean(np.stack(per_sec), 0)))


def train_fold(patient, tag, epochs=None, lr=None, batch_size=None, device=None,
               probe_every=0, resume=False):
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

    # optional calibration probe: held-out PCC every `probe_every` epochs
    probe_ds = TriTestSections(test_sections, panel) if probe_every else None
    probe_curve = []

    model = build_model().to(device)
    if config.FREEZE_TARGET:
        for p in model.target_encoder.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=config.WEIGHT_DECAY)

    ckpt_dir = os.path.join(config.CKPT_DIR, tag, f"fold_{patient}")
    os.makedirs(ckpt_dir, exist_ok=True)
    meta = dict(patient=patient, tag=tag, epochs=epochs, lr=lr, batch_size=batch_size,
                cpm=config.CPM, smooth=config.SMOOTH, freeze_target=config.FREEZE_TARGET,
                model_kwargs=config.MODEL_KWARGS, seed=config.SEED,
                train_sections=train_sections, test_sections=test_sections)

    start_ep = 0
    per_epoch = []
    resume_path = os.path.join(ckpt_dir, "resume.pt")
    if resume and os.path.isfile(resume_path):
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        if ck["meta"]["epochs"] != epochs:
            raise SystemExit(f"[fold {patient}] resume.pt was for {ck['meta']['epochs']} "
                             f"epochs, this run asks for {epochs}; refusing to mix budgets")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        _set_rng_state(ck["rng"])       # epoch-boundary RNG -> same shuffle/augment order
        start_ep = ck["epoch"]
        per_epoch = list(ck["per_epoch"])
        probe_curve = list(ck.get("probe_curve", []))
        print(f"[fold {patient}] RESUMED from epoch {start_ep}/{epochs} ({resume_path})")

    t0 = time.time()
    for ep in range(start_ep, epochs):
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
        msg = f"[fold {patient}] epoch {ep+1}/{epochs}  loss {avg:.4f}  {dt:.1f}s"
        if probe_every and ((ep + 1) % probe_every == 0 or ep == epochs - 1):
            r = _heldout_pcc(model, probe_ds, test_sections, device)
            probe_curve.append({"epoch": ep + 1, "heldout_pcc_mean": r})
            msg += f"  | held-out pcc {r:.4f}"
        if ep < 3 or (ep + 1) % 10 == 0 or ep == epochs - 1 or probe_curve:
            print(msg)

        done = ep + 1
        if config.SNAPSHOT_EVERY and done % config.SNAPSHOT_EVERY == 0:
            _atomic_save(dict(model=model.state_dict(), epoch=done, meta=meta),
                         os.path.join(ckpt_dir, f"epoch_{done:03d}.pt"))
        if config.CKPT_EVERY and done % config.CKPT_EVERY == 0 and done < epochs:
            _save_resume(ckpt_dir, model, opt, done, per_epoch, probe_curve, meta)

    # ---- final (scored) weights: last epoch, weights only ----
    _atomic_save(dict(model=model.state_dict(), epoch=epochs,
                      train_loss_curve=per_epoch, meta=meta),
                 os.path.join(ckpt_dir, "final.pt"))
    print(f"[fold {patient}] saved {os.path.join(ckpt_dir, 'final.pt')}")

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
    if probe_curve:
        run["heldout_probe"] = probe_curve
        with open(os.path.join(out_dir, "probe.json"), "w") as f:
            json.dump(probe_curve, f, indent=2)
        best = max(probe_curve, key=lambda x: x["heldout_pcc_mean"])
        print(f"[fold {patient}] held-out probe: peak pcc {best['heldout_pcc_mean']:.4f} "
              f"at epoch {best['epoch']}, last {probe_curve[-1]['heldout_pcc_mean']:.4f} "
              f"at epoch {probe_curve[-1]['epoch']}")
    run["checkpoint"] = os.path.join(ckpt_dir, "final.pt")
    run["resumed_from_epoch"] = start_ep
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run, f, indent=2)
    if os.path.isfile(resume_path):
        os.remove(resume_path)          # fold finished; final.pt is the keeper
    print(f"[fold {patient}] done in {run['seconds_total']/60:.1f} min -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--tag", default="triplex_lopo_833")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--probe_every", type=int, default=0,
                    help="log held-out PCC every N epochs (calibration only; "
                         "does not affect final last-epoch scoring)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from CKPT_DIR/<tag>/fold_<patient>/resume.pt if present")
    a = ap.parse_args()
    train_fold(a.patient, a.tag, epochs=a.epochs, lr=a.lr,
               batch_size=a.batch_size, probe_every=a.probe_every, resume=a.resume)
