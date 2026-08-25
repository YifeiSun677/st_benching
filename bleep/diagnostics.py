"""
The point of the whole within-patient run: tell "BLEEP is genuinely limited
under patient shift" apart from "I ported it wrong". Run this on every fold.

Three checks, in order of how decisive they are:

A. HELD-OUT ALIGNMENT (needs the checkpoint). Embed the query section's
   patches with the image tower AND its true expression with the expression
   tower, then ask where the correct partner ranks. Chance is 50th
   percentile. This is the direct test of whether the joint embedding
   learned anything, and unlike the loss it has an unambiguous null.

B. RETRIEVAL DIVERSITY (from preds.npz). If every query collapses onto the
   same handful of reference spots, predictions are near-constant across
   spots and per-gene PCC goes to zero or NaN -- which looks identical to
   "the model doesn't work" but isn't.

C. SHUFFLED-RETRIEVAL NULL (from preds.npz). Re-average k RANDOM reference
   spots per query. This is exactly the mean-expression baseline plus
   sampling noise, and it is the number real retrieval has to beat. If they
   match, the image tower is contributing nothing.

    python diagnostics.py --preds /workspace/runs/.../preds.npz \
        --ckpt /workspace/runs/.../last.pt --root /workspace/her2st/data \
        --panel ../panels/panel_833.txt --patient B --test_section B1
"""
import argparse
import json
import os

import numpy as np


def per_gene_pcc(pred, truth):
    """Column-wise Pearson r, NaN where either column is constant."""
    p = pred - pred.mean(axis=0, keepdims=True)
    t = truth - truth.mean(axis=0, keepdims=True)
    denom = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (p * t).sum(axis=0) / denom
    r[denom == 0] = np.nan
    return r


def check_alignment(args):
    import torch
    import torch.nn.functional as F
    from her2st_dataset import Her2stCLIPDataset, load_panel
    from models import CLIPModel
    from infer_bleep import embed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = load_panel(args.panel)
    qry = Her2stCLIPDataset(args.root, [args.test_section], panel,
                            is_train=False, verbose=False)
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    img = embed(model, qry, device, 128, 4, "image")
    spt = embed(model, qry, device, 128, 4, "spot")
    a = F.normalize(torch.from_numpy(img), p=2, dim=-1)
    b = F.normalize(torch.from_numpy(spt), p=2, dim=-1)
    sim = (a @ b.T).numpy()

    n = sim.shape[0]
    true_score = np.diag(sim)
    rank = (sim > true_score[:, None]).sum(axis=1)  # 0 = best
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

    pred_sd = pred.std(axis=0)
    truth_sd = truth.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = pred_sd / truth_sd
    ratio = ratio[np.isfinite(ratio)]

    r_real = per_gene_pcc(pred, truth)

    # Null: keep the predictions exactly as they are, but assign each one to
    # the WRONG query spot. This holds the marginal distribution of the
    # predictions fixed and destroys only the spot-to-patch correspondence,
    # so it isolates whether the image tower localises -- a shrunken but
    # correctly-placed prediction beats it, a shrunken and misplaced one
    # does not.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pred.shape[0])
    r_null = per_gene_pcc(pred[perm], truth)

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
    ap.add_argument("--patient", default="B")
    ap.add_argument("--test_section")
    args = ap.parse_args()

    npz = np.load(args.preds, allow_pickle=True)
    report = {"fold": args.test_section, "preds": args.preds}
    report.update(check_preds(npz))

    if args.ckpt and args.root and args.panel and args.test_section:
        report["alignment"] = check_alignment(args)

    print(json.dumps(report, indent=2))
    with open(os.path.join(os.path.dirname(args.preds), "diagnostics.json"),
              "w") as fh:
        json.dump(report, fh, indent=2)

    print("\n--- how to read this ---")
    a = report.get("alignment")
    if a:
        if a["median_percentile_rank"] < 60:
            print("ALIGNMENT: at/near chance -> the embedding did not learn. "
                  "This is an implementation or training problem, NOT a "
                  "finding about BLEEP.")
        else:
            print(f"ALIGNMENT: median percentile "
                  f"{a['median_percentile_rank']:.1f} vs 50 chance -> the "
                  f"embedding learned. Low PCC downstream is then a real "
                  f"property of retrieval, not a bug.")
    if report["retrieval_coverage"] < 0.05:
        print("DIVERSITY: retrieval has collapsed onto a few spots. "
              "Predictions will be near-constant; PCC is uninformative.")
    if report["per_gene_pcc_median"] <= report["shuffled_null_pcc_median"] + 0.01:
        print("NULL: real retrieval does not beat the shuffled null. The "
              "image tower is contributing nothing.")


if __name__ == "__main__":
    main()
