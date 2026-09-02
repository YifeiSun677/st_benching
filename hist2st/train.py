import argparse, json, os, random, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader
from . import config as C
from .her2st import list_sections, load_panel
from .dataset import Her2stHist2ST

sys.path.insert(0, str(C.HIST2ST_REPO))
from HIST2ST import Hist2ST      # noqa: E402


def build_folds(cv):
    secs = list_sections()
    if cv == "patient":                       # LOPO: primary protocol, 8 folds
        return [(p, [s for s in secs if s[0] != p], [s for s in secs if s[0] == p])
                for p in C.PATIENTS]
    if cv == "section":                       # reproduces the repo's samples = names[1:33]
        samples = secs[1:33]
        return [(s, [t for t in samples if t != s], [s]) for s in samples]
    raise ValueError(cv)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_opt(model, lr):
    o = model.configure_optimizers()
    if isinstance(o, (list, tuple)):
        o = o[0]
    if isinstance(o, dict):
        o = o["optimizer"]
    if o is None:
        o = torch.optim.Adam(model.parameters(), lr=lr)
    return o


def run_fold(args, fold_i, fold_name, tr, te, dev="cuda"):
    out = C.OUT_ROOT / args.tag / f"fold{fold_i:02d}_{fold_name}"
    (out / "preds").mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    panel = load_panel()
    ds_tr = Her2stHist2ST(tr, neighs=args.neighbor, prune=args.prune, scale255=args.scale255)
    ds_te = Her2stHist2ST(te, neighs=args.neighbor, prune=args.prune, scale255=args.scale255)
    dl_tr = DataLoader(ds_tr, batch_size=1, shuffle=True, num_workers=0)

    model = Hist2ST(depth1=args.depth1, depth2=args.depth2, depth3=args.depth3,
                    n_genes=len(panel), learning_rate=args.lr, label=None,
                    kernel_size=args.kernel, patch_size=args.patch, fig_size=C.PATCH,
                    heads=args.heads, channel=args.channel, dropout=args.dropout,
                    zinb=args.zinb, nb=False, bake=args.bake, lamb=args.lamb,
                    policy=args.policy).to(dev)
    model.log = lambda *a, **k: None
    model.log_dict = lambda *a, **k: None
    opt = get_opt(model, args.lr)

    torch.cuda.reset_peak_memory_stats()
    curve, t0 = [], time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for b in dl_tr:
            b = [x.to(dev) for x in b]
            o = model.training_step(b, 0)
            loss = o["loss"] if isinstance(o, dict) else o
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        curve.append(tot / len(dl_tr))
        if ep % args.log_every == 0 or ep == 1:
            print(f"  [{fold_name}] epoch {ep}/{args.epochs} loss {curve[-1]:.4f} "
                  f"({(time.time()-t0)/ep:.1f}s/ep)", flush=True)
    train_sec = time.time() - t0

    # ---- last-epoch scoring: no epoch selection of any kind ----
    model.eval()
    with torch.no_grad():
        for k, s in enumerate(te):
            patch, pos, exp, adj, ori, sf, ctr = ds_te[k]
            o = model(patch.unsqueeze(0).to(dev), pos.unsqueeze(0).to(dev), adj.to(dev))
            pred = o[0] if isinstance(o, (tuple, list)) else o
            pred = pred.squeeze(0) if pred.dim() == 3 else pred
            np.savez_compressed(out / "preds" / f"{s}.npz",
                                pred=pred.cpu().numpy().astype(np.float32),
                                truth=exp.numpy(), centers=ctr.numpy(),
                                spot_id=ds_te.data[s]["spot_id"], genes=np.array(panel))
    np.savetxt(out / "loss_curve.csv", np.array(curve), delimiter=",")
    if args.save_ckpt:
        torch.save(model.state_dict(), out / "model.pt")
    json.dump(dict(fold=fold_name, train=tr, test=te, config=vars(args),
                   n_genes=len(panel), train_seconds=round(train_sec, 1),
                   sec_per_epoch=round(train_sec / args.epochs, 2),
                   peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   final_train_loss=curve[-1], hist2st_commit=os.popen(
                       f"git -C {C.HIST2ST_REPO} rev-parse --short HEAD").read().strip()),
              open(out / "run.json", "w"), indent=2)
    print(f"  [{fold_name}] done in {train_sec/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hist2st_lopo_833")
    ap.add_argument("--cv", default="patient", choices=["patient", "section"])
    ap.add_argument("--folds", default="all", help="'all' or '0' or '0,3,5'")
    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--zinb", type=float, default=0.25)
    ap.add_argument("--bake", type=int, default=5)
    ap.add_argument("--lamb", type=float, default=0.5)
    ap.add_argument("--neighbor", type=int, default=4)
    ap.add_argument("--prune", default="Grid")
    ap.add_argument("--policy", default="mean")
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--patch", type=int, default=7)
    ap.add_argument("--depth1", type=int, default=2)
    ap.add_argument("--depth2", type=int, default=8)
    ap.add_argument("--depth3", type=int, default=4)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--channel", type=int, default=32)
    ap.add_argument("--scale255", action="store_true",
                    help="divide pixels by 255 (original repo does not; off by default)")
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    folds = build_folds(args.cv)
    sel = range(len(folds)) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    for i in sel:
        name, tr, te = folds[i]
        print(f"=== fold {i} ({name}): train {len(tr)} sections / test {len(te)} ===", flush=True)
        run_fold(args, i, name, tr, te)


if __name__ == "__main__":
    main()
