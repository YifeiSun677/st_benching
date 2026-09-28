"""
Per-gene Gaussian prior for the STFlow normalised-target run (option A).

STFlow's upstream prior is ZINB, a count distribution. The new target is
continuous and non-integer, so ZINB cannot be used unchanged. This replaces it
with the simplest matched substitution: an independent Gaussian per gene, with
mean and sd estimated from the TRAINING patients of that fold only.

Contract:
  - fit() sees training-patient spots only. Held-out patients never touch it.
  - sample() draws on GPU when the prior has been .to('cuda'), preserving the
    `--prior_on_gpu` behaviour of the old run (8.2 vs 20.2 s/epoch).
  - save() writes mu/sd/genes so the fold's prior is auditable after the fact.

Note on sign: Gaussian samples can be negative while the target is >= 0. That is
fine for flow matching (x_t is an interpolation, not a count), but if any part of
the port assumes a non-negative x0, set clip_nonneg=True and say so in methods.
Default is off.
"""
from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np

try:
    import torch
except ImportError:  # allows preflight to run on a CPU-only box without torch
    torch = None


DEFAULT_SD_FLOOR = 1e-3


class GaussianPrior:
    """x0_g ~ N(mu_g, sd_g), independent across genes."""

    def __init__(self, mu: np.ndarray, sd: np.ndarray,
                 genes: Optional[Sequence[str]] = None,
                 meta: Optional[dict] = None):
        mu = np.asarray(mu, dtype=np.float32).reshape(-1)
        sd = np.asarray(sd, dtype=np.float32).reshape(-1)
        if mu.shape != sd.shape:
            raise ValueError(f"mu {mu.shape} and sd {sd.shape} disagree")
        if not np.isfinite(mu).all() or not np.isfinite(sd).all():
            raise ValueError("prior parameters contain NaN/Inf")
        if (sd <= 0).any():
            raise ValueError("prior sd must be strictly positive (apply sd_floor)")
        self.mu = mu
        self.sd = sd
        self.genes = list(genes) if genes is not None else None
        self.meta = dict(meta or {})
        self._t_mu = None
        self._t_sd = None
        self._device = "cpu"

    # ---------------------------------------------------------------- fitting

    @classmethod
    def fit(cls, y_train: np.ndarray, sd_floor: float = DEFAULT_SD_FLOOR,
            genes: Optional[Sequence[str]] = None,
            meta: Optional[dict] = None) -> "GaussianPrior":
        """y_train: (n_train_spots, n_genes) target matrix, training patients only."""
        y = np.asarray(y_train, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"y_train must be 2-D; got {y.shape}")
        if y.shape[0] < 2:
            raise ValueError("need at least 2 spots to estimate a per-gene sd")
        if not np.isfinite(y).all():
            raise ValueError("y_train contains NaN/Inf")

        mu = y.mean(axis=0)
        sd_raw = y.std(axis=0, ddof=0)          # population sd; ddof is irrelevant at n~10^4
        n_floored = int((sd_raw < sd_floor).sum())
        sd = np.maximum(sd_raw, sd_floor)

        info = {
            "n_train_spots": int(y.shape[0]),
            "n_genes": int(y.shape[1]),
            "sd_floor": float(sd_floor),
            "n_genes_at_sd_floor": n_floored,
            "mu_min": float(mu.min()), "mu_max": float(mu.max()),
            "sd_min": float(sd.min()), "sd_max": float(sd.max()),
            "sd_median": float(np.median(sd)),
        }
        info.update(meta or {})
        return cls(mu, sd, genes=genes, meta=info)

    # --------------------------------------------------------------- sampling

    def to(self, device) -> "GaussianPrior":
        if torch is None:
            raise RuntimeError("torch not available")
        self._device = str(device)
        self._t_mu = torch.as_tensor(self.mu, dtype=torch.float32, device=device)
        self._t_sd = torch.as_tensor(self.sd, dtype=torch.float32, device=device)
        return self

    def sample(self, n: int, device=None, generator=None,
               dtype=None, clip_nonneg: bool = False):
        """Draw (n, n_genes) prior samples. Drop-in for the ZINB sampler."""
        if torch is None:
            raise RuntimeError("torch not available")
        if device is not None and str(device) != self._device:
            self.to(device)
        if self._t_mu is None:
            self.to("cpu")
        dev = self._t_mu.device
        dtype = dtype or torch.float32
        eps = torch.randn((n, self._t_mu.numel()), generator=generator,
                          device=dev, dtype=dtype)
        x0 = self._t_mu.to(dtype) + self._t_sd.to(dtype) * eps
        if clip_nonneg:
            x0 = x0.clamp_min_(0.0)
        return x0

    def sample_like(self, x, generator=None, clip_nonneg: bool = False):
        """Same shape/device/dtype as an existing batch tensor x."""
        if x.shape[-1] != self.mu.shape[0]:
            raise ValueError(
                f"batch has {x.shape[-1]} gene columns, prior has {self.mu.shape[0]}")
        return self.sample(x.shape[0], device=x.device, generator=generator,
                           dtype=x.dtype, clip_nonneg=clip_nonneg)

    # ------------------------------------------------------------------- i/o

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, mu=self.mu, sd=self.sd,
            genes=np.array(self.genes if self.genes is not None else [], dtype=object),
            meta=json.dumps(self.meta))
        with open(str(path).replace(".npz", ".json"), "w") as fh:
            json.dump(self.meta, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "GaussianPrior":
        z = np.load(path, allow_pickle=True)
        genes = list(z["genes"]) if z["genes"].size else None
        meta = json.loads(str(z["meta"]))
        return cls(z["mu"], z["sd"], genes=genes, meta=meta)

    def summary(self) -> dict:
        return dict(self.meta)

    def __repr__(self) -> str:
        return (f"GaussianPrior(n_genes={self.mu.shape[0]}, "
                f"sd_median={np.median(self.sd):.4f}, device={self._device})")


def build_prior(kind: str, y_train: np.ndarray, genes=None,
                sd_floor: float = DEFAULT_SD_FLOOR, **kw):
    """Dispatch used by train(). 'zinb' is intentionally left to the existing
    upstream code path so the old run can still be reproduced."""
    if kind == "gaussian":
        return GaussianPrior.fit(y_train, sd_floor=sd_floor, genes=genes, meta=kw)
    raise ValueError(
        f"build_prior only handles 'gaussian'; got {kind!r}. "
        "Keep the ZINB branch on the upstream sampler.")
