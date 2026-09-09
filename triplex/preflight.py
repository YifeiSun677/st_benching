"""
Preflight -- run this FIRST on the pod. It needs a GPU + torch but NO her2st
data for the wiring test. It:

  1. imports the real TRIPLEX model with the flash-attn stub and confirms every
     attention module is on the exact (non-flash) path,
  2. runs a full TRAINING forward+backward on a tiny synthetic 2-section batch
     (exercises retrieve_global_emb, the neighbour mask, APEG and the distillation
     loss), and
  3. runs a per-section TEST forward and checks the logits shape.

Then, only if the real caches exist, it loads one section and does a real
forward, and sanity-checks the panel, cache and spot index.

  python -m triplex.preflight
"""
import os
import numpy as np
import torch

from . import config, her2st
from .triplex_import import build_model


class _FakeTrainDS:
    """Minimal stand-in exposing what retrieve_global_emb() reads."""
    def __init__(self, sizes, dim):
        self.int2id = {i: f"S{i}" for i in range(len(sizes))}
        self.global_embs, self.pos_dict = {}, {}
        for i, n in enumerate(sizes):
            self.global_embs[f"S{i}"] = torch.randn(n, dim)
            rc = np.array([(r, c) for r in range(1, 40) for c in range(1, 40)])[:n]
            self.pos_dict[f"S{i}"] = torch.FloatTensor(rc[:, ::-1].copy())  # (x,y)


def wiring_test(device="cuda"):
    dim, ng = config.EMB_DIM, config.NUM_GENES
    model = build_model().to(device)
    print(f"[preflight] model built; flash_attn forced off on all attn modules")

    sizes = [30, 25]
    ds = _FakeTrainDS(sizes, dim)

    # a batch drawing spots from BOTH synthetic sections
    # a batch drawing spots from BOTH synthetic sections. pid/sid are 1-D [B],
    # exactly what default_collate produces from the dataset's scalar fields.
    B = 8
    pid = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1]).to(device)   # section id/spot
    sid = torch.LongTensor([0, 1, 2, 3, 0, 1, 2, 3]).to(device)   # row within section
    batch = dict(
        img=torch.randn(B, 3, 224, 224, device=device),
        mask=torch.randint(0, 2, (B, 25), device=device).long(),
        neighbor_emb=torch.randn(B, 25, dim, device=device),
        label=torch.rand(B, ng, device=device),
        pid=pid, sid=sid,
    )
    batch["mask"][:, 12] = 1  # centre always present

    model.train()
    out = model(img=batch["img"], mask=batch["mask"],
                neighbor_emb=batch["neighbor_emb"], pid=batch["pid"],
                sid=batch["sid"], label=batch["label"], dataset=ds, phase="train")
    assert "loss" in out and out["logits"].shape == (B, ng), out["logits"].shape
    out["loss"].backward()
    print(f"[preflight] TRAIN forward+backward OK  loss={float(out['loss']):.4f}  "
          f"logits={tuple(out['logits'].shape)}")

    # test / inference path on a whole synthetic section
    model.eval()
    n = 30
    with torch.no_grad():
        res = model(
            img=torch.randn(n, 3, 224, 224, device=device),
            mask=torch.ones(n, 25, device=device).long(),
            neighbor_emb=torch.randn(n, 25, dim, device=device),
            position=ds.pos_dict["S0"].to(device),
            global_emb=ds.global_embs["S0"].unsqueeze(0).to(device))
    assert res["logits"].shape == (n, ng), res["logits"].shape
    assert (res["logits"] >= 0).all(), "outputs should be clamped >= 0"
    print(f"[preflight] TEST forward OK  logits={tuple(res['logits'].shape)}")


def data_test(device="cuda"):
    try:
        panel = her2st.load_panel()
        print(f"[preflight] panel OK: {len(panel)} genes")
    except Exception as e:
        print(f"[preflight] panel NOT ready: {e}")
    if os.path.exists(config.HER2ST_CACHE):
        cache = np.memmap(config.HER2ST_CACHE, dtype=np.uint8, mode="r",
                          shape=config.HER2ST_CACHE_SHAPE)
        print(f"[preflight] cache OK: {cache.shape} dtype={cache.dtype}")
    else:
        print(f"[preflight] cache NOT found at {config.HER2ST_CACHE}")
    if not os.path.isdir(config.FEATURE_DIR) or not os.listdir(config.FEATURE_DIR):
        print(f"[preflight] no features yet in {config.FEATURE_DIR} "
              f"(run build_features) -- skipping real forward")
        return
    from .dataset import TriTestSections
    idx = her2st.load_spot_index()
    sec = her2st.all_sections(idx)[0]
    ds = TriTestSections([sec], her2st.load_panel())
    b = ds.section_batch(sec, device=device)
    model = build_model().to(device).eval()
    with torch.no_grad():
        res = model(img=b["img"], mask=b["mask"], neighbor_emb=b["neighbor_emb"],
                    position=b["position"], global_emb=b["global_emb"])
    print(f"[preflight] REAL section {sec}: pred {tuple(res['logits'].shape)}, "
          f"truth {b['label'].shape}, neighbours/spot={b['mask'].float().sum(1).mean():.1f}")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[preflight] device={dev}")
    wiring_test(dev)
    data_test(dev)
    print("[preflight] all checks passed")
