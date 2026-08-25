"""
BLEEP inference for one fold. Replaces the hardcoded notebook.

Reference set = the TRAINING sections only. Getting this wrong (e.g.
including the query section, as the published notebook's
build_loaders_inference does before slicing) silently inflates everything.

Saves preds.npz with:
    pred      (n_query, n_genes)  top-k averaged reference expression
    truth     (n_query, n_genes)  CPM log1p ground truth
    baseline  (n_genes,)          per-gene mean of the reference
    indices   (n_query, k)        which reference spots were retrieved
    genes, query_keys, ref_keys
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from her2st_dataset import Her2stCLIPDataset, load_panel, sections_for_patient
from models import CLIPModel


@torch.no_grad()
def embed(model, dataset, device, batch_size, num_workers, what):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    out = []
    model.eval()
    for batch in loader:
        if what == "image":
            out.append(model.embed_image(batch["image"].to(device)).cpu())
        else:
            out.append(model.embed_spot(
                batch["reduced_expression"].to(device)).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--patient", default="B")
    ap.add_argument("--test_section", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--method", default="average",
                    choices=["average", "weighted_average", "simple"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = load_panel(args.panel)

    all_sections = sections_for_patient(args.root, args.patient)
    train_sections = [s for s in all_sections if s != args.test_section]
    print(f"query={args.test_section}  reference={train_sections}")

    # is_train=False everywhere: no augmentation at inference, matching the
    # benchmark-wide no-TTA rule.
    ref_ds = Her2stCLIPDataset(args.root, train_sections, panel, is_train=False)
    qry_ds = Her2stCLIPDataset(args.root, [args.test_section], panel,
                               is_train=False)

    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    print("embedding reference spots (expression tower)...")
    ref_emb = embed(model, ref_ds, device, args.batch_size, args.num_workers, "spot")
    print("embedding query patches (image tower)...")
    qry_emb = embed(model, qry_ds, device, args.batch_size, args.num_workers, "image")

    ref_expr = ref_ds.expression_matrix()
    qry_expr = qry_ds.expression_matrix()

    q = F.normalize(torch.from_numpy(qry_emb), p=2, dim=-1)
    r = F.normalize(torch.from_numpy(ref_emb), p=2, dim=-1)
    sim = q @ r.T                                    # (n_query, n_ref)
    k = min(args.top_k, r.shape[0])
    _, idx = torch.topk(sim, k=k, dim=-1)
    idx = idx.numpy()

    if args.method == "simple":
        pred = ref_expr[idx[:, 0], :]
        idx = idx[:, :1]
    elif args.method == "average":
        pred = ref_expr[idx].mean(axis=1)
    else:  # weighted_average, BLEEP's exp(-(d^2 - d_min^2 + 1)) scheme
        qn, rn = q.numpy(), r.numpy()
        pred = np.zeros((idx.shape[0], ref_expr.shape[1]), dtype=np.float32)
        for i in range(idx.shape[0]):
            d = ((rn[idx[i]] - qn[i]) ** 2).sum(axis=1)
            w = np.exp(-(d - d.min() + 1))
            pred[i] = np.average(ref_expr[idx[i]], axis=0, weights=w)

    np.savez_compressed(
        os.path.join(args.out, "preds.npz"),
        pred=pred.astype(np.float32),
        truth=qry_expr.astype(np.float32),
        baseline=ref_expr.mean(axis=0).astype(np.float32),
        indices=idx.astype(np.int32),
        genes=np.array(panel),
        query_keys=np.array(qry_ds.spot_keys()),
        ref_keys=np.array(ref_ds.spot_keys()),
        method=np.array(args.method),
        top_k=np.array(k),
    )
    print(f"saved {pred.shape} -> {args.out}/preds.npz")


if __name__ == "__main__":
    main()
