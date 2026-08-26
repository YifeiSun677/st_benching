"""
BLEEP inference for one fold. Reference set = TRAINING sections only.

Same dual interface as train_bleep.py:
  --patient B --test_section B1
  --train_sections A1,A2,... --test_sections B1,B2,...

Saves preds.npz with pred / truth / baseline / indices / genes / keys.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import splits
from her2st_dataset import load_panel
from models import CLIPModel
from patch_cache import build_dataset


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


def resolve_split(args):
    if args.train_sections and args.test_sections:
        return (splits.parse_sections(args.train_sections),
                splits.parse_sections(args.test_sections))
    if args.patient and args.test_section:
        return splits.loso_split(args.root, args.patient, args.test_section)
    raise SystemExit("give --train_sections/--test_sections, or "
                     "--patient/--test_section")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--patient")
    ap.add_argument("--test_section")
    ap.add_argument("--train_sections")
    ap.add_argument("--test_sections")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--method", default="average",
                    choices=["average", "weighted_average", "simple"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = load_panel(args.panel)

    train_sections, test_sections = resolve_split(args)
    print(f"query {test_sections}  reference {len(train_sections)} sections")

    ref_ds = build_dataset(args, train_sections, panel, is_train=False)
    qry_ds = build_dataset(args, test_sections, panel, is_train=False)

    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    print("embedding reference spots...")
    ref_emb = embed(model, ref_ds, device, args.batch_size, args.num_workers, "spot")
    print("embedding query patches...")
    qry_emb = embed(model, qry_ds, device, args.batch_size, args.num_workers, "image")

    ref_expr = ref_ds.expression_matrix()
    qry_expr = qry_ds.expression_matrix()

    q = F.normalize(torch.from_numpy(qry_emb), p=2, dim=-1)
    r = F.normalize(torch.from_numpy(ref_emb), p=2, dim=-1)
    sim = q @ r.T
    k = min(args.top_k, r.shape[0])
    _, idx = torch.topk(sim, k=k, dim=-1)
    idx = idx.numpy()

    if args.method == "simple":
        pred = ref_expr[idx[:, 0], :]
        idx = idx[:, :1]
    elif args.method == "average":
        pred = ref_expr[idx].mean(axis=1)
    else:
        qn, rn = q.numpy(), r.numpy()
        pred = np.zeros((idx.shape[0], ref_expr.shape[1]), dtype=np.float32)
        for i in range(idx.shape[0]):
            d = ((rn[idx[i]] - qn[i]) ** 2).sum(axis=1)
            w = np.exp(-(d - d.min() + 1))
            pred[i] = np.average(ref_expr[idx[i]], axis=0, weights=w)

    np.savez_compressed(
        os.path.join(args.out, "preds.npz"),
        pred=pred.astype(np.float32), truth=qry_expr.astype(np.float32),
        baseline=ref_expr.mean(axis=0).astype(np.float32),
        indices=idx.astype(np.int32), genes=np.array(panel),
        query_keys=np.array(qry_ds.spot_keys()),
        ref_keys=np.array(ref_ds.spot_keys()),
        method=np.array(args.method), top_k=np.array(k))
    print(f"saved {pred.shape} -> {args.out}/preds.npz")


if __name__ == "__main__":
    main()
