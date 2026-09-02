"""Every check to run before the real training. Writes nothing."""
import sys, numpy as np, torch
from . import config as C
from .her2st import list_sections, load_section, load_panel
from .dataset import Her2stHist2ST

sys.path.insert(0, str(C.HIST2ST_REPO))
from HIST2ST import Hist2ST      # noqa: E402


def main():
    panel = load_panel()
    secs = list_sections()
    print(f"[1] sections={len(secs)} panel={len(panel)} norm={C.NORM}")
    assert len(secs) == 36, secs

    tot, miss = 0, []
    for s in secs:
        d = load_section(s, panel)
        tot += len(d["spot_id"])
        miss.append(d["n_missing"])
    print(f"[2] spots={tot} (expect 13620) | zero-filled genes per section: "
          f"min {min(miss)} / median {int(np.median(miss))} / max {max(miss)}")

    ds = Her2stHist2ST(["B1"])
    patch, pos, exp, adj, ori, sf, ctr = ds[0]
    print(f"[3] shapes patch{tuple(patch.shape)} pos{tuple(pos.shape)} exp{tuple(exp.shape)} "
          f"adj{tuple(adj.shape)} ori{tuple(ori.shape)} sf{tuple(sf.shape)} center{tuple(ctr.shape)}")
    assert patch.shape[1:] == (3, 112, 112) and exp.shape[1] == len(panel)

    print(f"[4] position range x[{pos[:,0].min()},{pos[:,0].max()}] y[{pos[:,1].min()},{pos[:,1].max()}] "
          f"(must be < 64 or the pos-embedding overflows)")
    assert pos.max() < 64

    deg = adj.sum(1)
    print(f"[5] adj degree mean {deg.mean():.2f} min {deg.min():.0f} max {deg.max():.0f} "
          f"| symmetric fraction {(adj == adj.T).float().mean():.3f}")

    print(f"[6] patch pixels min {patch.min():.1f} max {patch.max():.1f} mean {patch.mean():.1f} "
          f"| exp min {exp.min():.3f} max {exp.max():.3f} | ori integral {bool((ori % 1 == 0).all())}")

    m = Hist2ST(depth1=2, depth2=8, depth3=4, n_genes=len(panel), learning_rate=1e-5,
                kernel_size=5, patch_size=7, fig_size=112, heads=16, channel=32,
                dropout=0.2, zinb=0.25, nb=False, bake=5, lamb=0.5, policy='mean').cuda()
    b = [x.unsqueeze(0).cuda() for x in (patch, pos, exp, adj, ori, sf, ctr)]
    m.log = lambda *a, **k: None
    m.log_dict = lambda *a, **k: None
    out = m.training_step(b, 0)
    loss = out["loss"] if isinstance(out, dict) else out
    loss.backward()
    print(f"[7] training_step OK, loss={float(loss):.4f}, "
          f"peak GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GB")

    m.eval()
    with torch.no_grad():
        o = m(b[0], b[1], b[3].squeeze(0))
        pred = o[0] if isinstance(o, (tuple, list)) else o
    print(f"[8] forward OK, pred {tuple(pred.shape)}")


if __name__ == "__main__":
    main()
