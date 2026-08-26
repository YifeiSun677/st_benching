"""
Learned-vs-broken checks. Same three as the control run:

A. HELD-OUT ALIGNMENT (needs the checkpoint). Embed the query sections'
   patches with the image tower AND their true expression with the
   expression tower; ask where the correct partner ranks. Chance is the
   50th percentile.
B. RETRIEVAL DIVERSITY. Collapse onto a few reference spots makes
   predictions near-constant and PCC uninformative rather than bad.
C. PERMUTATION NULL. Predictions kept as-is but assigned to the wrong
   query spot: isolates whether the image tower localises.
"""
import argparse
import json
import os

import numpy as np

import splits


def per_gene_pcc(pred, truth):
    p = pred - pred.mean(axis=0, keepdims=True)
    t = truth - truth.mean(axis=0, keepdims=True)
    denom = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (p * t).sum(axis=0) / denom
    r[denom == 0] = np.nan
    return r


def check_alignment(args, test_sections):
    import torch
    import torch.nn.functional as F
    from her2st_dataset import load_panel
    from models import CLIPModel
    from infer_bleep import embed
    from patch_cache import build_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = load_panel(args.panel)
    qry = build_dataset(args, test_sections, panel, is_train=False,
                        verbose=False)
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    img = embed(model, qry, device, 128, 4, "image")
    spt = embed(model, qry, device, 128, 4, "spot")
    a = F.normalize(torch.from_numpy(img), p=2, dim=-1)
    b = F.normalize(torch.from_numpy(spt), p=2, dim=-1)
    sim = (a @ b.T).numpy()

    n = sim.shape[0]
    true_score = np.diag(sim)
    rank = (sim > true_score[:, None]).sum(axis=1)
    pct = 100 * (1 - rank / max(n - 1, 1))
    return {
        "n_query_spots": int(n),
        "top1_accuracy": float((rank == 0).mean()),
        "top10_accuracy": float((rank < 10).mean()),
        "median_percentile_rank": float(np.median(pct)),
        "chance_percentile_rank": 50.0,
    }


def check_preds(npz, seed=0):
    pred, truth = npz["pred"], npz["truth"]
    idx, ref_keys = npz["indices"], npz["ref_keys"]
    n_ref, k = len(ref_keys), idx.shape[1]

    counts = np.bincount(idx.ravel(), minlength=n_ref)
    used = int((counts > 0).sum())
    top10_share = float(np.sort(counts)[::-1][:10].sum() / counts.sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = pred.std(axis=0) / truth.std(axis=0)
    ratio = ratio[np.isfinite(ratio)]

    r_real = per_gene_pcc(pred, truth)
    rng = np.random.default_rng(seed)
    r_null = per_gene_pcc(pred[rng.permutation(pred.shape[0])], truth)

    return {
        "n_reference_spots": int(n_ref),
        "top_k": int(k),
        "unique_reference_spots_retrieved": used,
        "retrieval_coverage": float(used / n_ref),
        "share_of_retrievals_from_top10_spots": top10_share,
        "median_pred_sd_over_truth_sd": float(np.median(ratio)),
        "per_gene_pcc_median": float(np.nanmedian(r_real)),
        "per_gene_pcc_mean": float(np.nanmean(r_real)),
        "per_gene_pcc_nan_count": int(np.isnan(r_real).sum()),
        "shuffled_null_pcc_median": float(np.nanmedian(r_null)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--root")
    ap.add_argument("--panel")
    ap.add_argument("--cache")
    ap.add_argument("--patient")
    ap.add_argument("--test_section")
    ap.add_argument("--test_sections")
    args = ap.parse_args()

    npz = np.load(args.preds, allow_pickle=True)
    report = {"preds": args.preds}
    report.update(check_preds(npz))

    test_sections = None
    if args.test_sections:
        test_sections = splits.parse_sections(args.test_sections)
    elif args.test_section:
        test_sections = [args.test_section]
    report["test_sections"] = test_sections

    if args.ckpt and args.panel and test_sections:
        report["alignment"] = check_alignment(args, test_sections)

    print(json.dumps(report, indent=2))
    with open(os.path.join(os.path.dirname(args.preds),
                           "diagnostics.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    a = report.get("alignment")
    if a and a["median_percentile_rank"] < 60:
        print("\nALIGNMENT at/near chance -> embedding did not learn.")
    if report["retrieval_coverage"] < 0.05:
        print("\nDIVERSITY: retrieval collapsed; PCC is uninformative.")
    if report["per_gene_pcc_median"] <= report["shuffled_null_pcc_median"] + 0.01:
        print("\nNULL: real retrieval does not beat the permutation null.")


if __name__ == "__main__":
    main()
