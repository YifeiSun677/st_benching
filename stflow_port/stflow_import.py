"""
stflow_port/stflow_import.py -- load the model and flow code from a READ-ONLY clone
of Graph-and-Geometric-Learning/STFlow and apply the minimum fixes needed to run it.

Why a shim instead of `pip install -e .`:
  * the upstream package is called `stflow`; our folder is `stflow_port` precisely so
    the two never collide (same lesson as path2space/p2s_companion);
  * upstream `stflow/data/normalize_utils.py` imports `scprep` at module top (pins
    pandas<2.1, fails on py3.12) and `stflow/utils` imports `mygene`; the ZINB prior
    imports `scvi-tools`. We import none of those modules, so none are installed.

Fixes applied (all verified by preflight.py):
  FIX 1  TransformerBlock passes `non_negative=` to GeneUpdate, whose __init__ does not
         accept it -> the released code raises TypeError before training starts.
         We accept and ignore the kwarg (the released GeneUpdate never implemented it).
  FIX 2  MLPAttnEdgeAggregation hard-codes the attention-MLP input width as
         d_head*2 + d_edge_head + 50, i.e. it only works for a 50-gene panel (HEST-bench's
         var_50genes). With 833 genes the forward pass fails with a shape error
         (929 vs 146). We rebuild mlp_attn with `+ n_genes`. At n_genes=50 this is
         an exact identity with the released architecture (preflight checks it).
  FIX 3  FrameAveraging.create_ops indexes a tensor with a Python list containing
         None/slice (deprecated; a future torch will reinterpret it). Replaced with the
         identical tuple index. Output compared with the original in preflight.
  PRIOR  scvi's ZeroInflatedNegativeBinomial(total_count, logits, zi_logits).sample is
         re-implemented with torch.distributions using scvi's exact parametrisation
         (theta = total_count, mu = exp(logits)*theta, Gamma(theta, theta/mu) -> Poisson,
         zero w.p. sigmoid(zi_logits)). Moments checked analytically in preflight.
"""
import os
import sys
import warnings

import config as C

_LOADED = {}


def _add_clone_to_path():
    repo = C.STFLOW_REPO
    if not os.path.isdir(os.path.join(repo, "stflow", "model")):
        raise FileNotFoundError(
            f"STFlow clone not found at {repo} (expected {repo}/stflow/model). "
            f"Clone it: git clone https://github.com/Graph-and-Geometric-Learning/STFlow.git {repo}")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import stflow  # namespace package (upstream ships no __init__.py)
    paths = [os.path.realpath(p) for p in list(stflow.__path__)]
    bad = [p for p in paths if not p.startswith(os.path.realpath(repo))]
    if bad:
        raise RuntimeError(f"`stflow` resolves outside the clone: {bad}. "
                           f"Is there another folder called stflow on sys.path / cwd?")
    return repo


def _patch_transformer(T):
    import torch.nn as nn
    from timm.models.vision_transformer import Mlp, SwiGLUPacked

    # FIX 1 ---------------------------------------------------------------
    # (patch __init__ in place; subclassing would break upstream's
    #  `super(GeneUpdate, self).__init__()`, which looks the class up by global name)
    if not getattr(T.GeneUpdate.__init__, "_stflow_port_fixed", False):
        _orig_gu_init = T.GeneUpdate.__init__

        def _gu_init(self, d_model, n_genes, proj_drop=0., non_negative=True, **kw):
            _orig_gu_init(self, d_model, n_genes, proj_drop=proj_drop)

        _gu_init._stflow_port_fixed = True
        T.GeneUpdate.__init__ = _gu_init

    # FIX 2 ---------------------------------------------------------------
    if not getattr(T.TransformerBlock.__init__, "_stflow_port_fixed", False):
        _orig_init = T.TransformerBlock.__init__

        def _init(self, d_model, d_edge_model, n_genes, n_heads=1, activation="gelu",
                  attn_drop=0., proj_drop=0., gene_exp_non_negative=True, mlp_ratio=4.0):
            _orig_init(self, d_model, d_edge_model, n_genes, n_heads=n_heads,
                       activation=activation, attn_drop=attn_drop, proj_drop=proj_drop,
                       gene_exp_non_negative=gene_exp_non_negative, mlp_ratio=mlp_ratio)
            a = self.attn
            in_f = a.d_head * 2 + a.d_edge_head + n_genes
            if activation == "swiglu":
                a.mlp_attn = SwiGLUPacked(in_features=in_f, hidden_features=d_model,
                                          out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm)
            else:  # upstream non-swiglu branch uses timm Mlp with its default GELU
                a.mlp_attn = Mlp(in_features=in_f, hidden_features=d_model,
                                 out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm)
            a._stflow_port_attn_in = in_f

        _init._stflow_port_fixed = True
        T.TransformerBlock.__init__ = _init


def _patch_fa(FA):
    import torch
    from einops import rearrange

    if getattr(FA.FrameAveraging.create_ops, "_stflow_port_fixed", False):
        return
    _FAorig_create_ops = FA.FrameAveraging.create_ops

    def create_ops(self, dim):
        colon = slice(None)
        accum = []
        directions = torch.tensor([-1, 1])
        for ind in range(dim):
            dim_slice = [None] * dim
            dim_slice[ind] = colon
            accum.append(directions[tuple(dim_slice)])
        accum = torch.broadcast_tensors(*accum)
        operations = torch.stack(accum, dim=-1)
        return rearrange(operations, '... d -> (...) d')

    create_ops._stflow_port_fixed = True
    FA.FrameAveraging.create_ops = create_ops
    FA.FrameAveraging._original_create_ops = _FAorig_create_ops


def make_zinb_sampler(total_count, logits, zi_logits, device="cpu"):
    """Exact re-implementation of
    scvi.distributions.ZeroInflatedNegativeBinomial(total_count=tensor([tc]),
    logits=tensor([lg]), zi_logits=zl).sample(shape).squeeze(-1)  (scvi-tools 1.5).
    device='cpu' reproduces upstream (scvi samples on the CPU because total_count is a CPU
    tensor); device='cuda' draws from the identical distribution on the GPU (speed lever)."""
    import torch
    theta = torch.tensor([float(total_count)], device=device)
    mu = torch.exp(torch.tensor([float(logits)], device=device)) * theta
    pi = torch.sigmoid(torch.tensor([float(zi_logits)], device=device))
    gamma = torch.distributions.Gamma(concentration=theta, rate=theta / mu)

    def sample(shape):
        with torch.no_grad():
            p_means = gamma.sample(torch.Size(shape))          # shape + (1,)
            counts = torch.poisson(torch.clamp(p_means, max=1e8))
            is_zero = torch.rand_like(counts) <= pi
            counts = torch.where(is_zero, torch.zeros_like(counts), counts)
        return counts.squeeze(-1)

    sample.analytic = dict(
        mean=float(((1 - pi) * mu).cpu()),
        var=float(((1 - pi) * mu * (mu + theta + pi * mu * theta) / theta).cpu()),
        p_zero=float((pi + (1 - pi) * (theta / (theta + mu)) ** theta).cpu()),
    )
    return sample


def load():
    """Returns a dict of upstream objects (imported once, patched once)."""
    if _LOADED:
        return _LOADED
    repo = _add_clone_to_path()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import stflow.model.fa as FA
        _patch_fa(FA)
        import stflow.model.transformer as T
        _patch_transformer(T)
        from stflow.model.denoiser import Denoiser
        from stflow.flow.interpolant import Interpolant
        from stflow.data.sampling_utils import PatchSampler
        from stflow.data.dataset import SPData, padding_batcher
        from stflow.app.flow.test import test as upstream_test, metric_func
    _LOADED.update(repo=repo, FA=FA, T=T, Denoiser=Denoiser, Interpolant=Interpolant,
                   PatchSampler=PatchSampler, SPData=SPData, padding_batcher=padding_batcher,
                   upstream_test=upstream_test, metric_func=metric_func)
    return _LOADED


def build_interpolant(args):
    """Upstream Interpolant with the ZINB prior swapped in (no scvi dependency).
    Upstream: Interpolant(prior, total_count=..., logits=..., zi_logits=...,
                          normalize = prior != 'gaussian')."""
    U = load()
    if args.prior_sampler != "zinb":
        return U["Interpolant"](args.prior_sampler, normalize=args.prior_sampler != "gaussian")
    interp = U["Interpolant"]("zero", normalize=True)
    dev = "cuda" if getattr(args, "prior_on_gpu", False) else "cpu"
    fn = make_zinb_sampler(args.zinb_total_count, args.zinb_logits, args.zinb_zi_logits, dev)
    interp.prior_sampler.prior_sampler = fn
    interp.prior_sampler.prior_sample_type = "zinb"
    return interp


def git_commit(path):
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
