"""
Latent Action Model with a swappable bottleneck.

Two branches share everything except the bottleneck module:

   Discrete branch (LAPO / Genie / GO-1 family):
       VQBottleneck — VQ-VAE codebook with straight-through estimator

   Continuous branch (DreamDojo / AdaWorld / GR00T family):
       GaussianBottleneck — KL-regularized Gaussian with reparam trick + free bits

Architecture (held fixed across both branches):

    encoder (frozen DINOv2-small, 22M params)  ->  features f_t in R^384
    IDM:        (f_t, f_{t+1})  ->  z_pre in R^d
    bottleneck:        z_pre    ->  z_used   (z_q for VQ, sampled z for Gaussian)
                                  + z_eval  (z_q for VQ, mu for Gaussian)  [for probes]
    FDM:        (f_t, z_used)   ->  predicted f_{t+1}

Loss = MSE(f_pred, f_{t+1}) + bottleneck_loss
       (commitment loss for VQ; β·KL with free bits for Gaussian)
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_dinov2_small() -> nn.Module:
    """Frozen DINOv2-small. CLS token, 384-dim."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Bottlenecks
# =============================================================================

class VQBottleneck(nn.Module):
    """VQ-VAE bottleneck with straight-through estimator (LAPO-style)."""

    branch_name = "vq"

    def __init__(self, dim: int = 64, num_codes: int = 64, beta: float = 0.25):
        super().__init__()
        self.dim = dim
        self.num_codes = num_codes
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z: torch.Tensor) -> dict[str, Any]:
        # z: (N, dim)
        d = (
            z.pow(2).sum(-1, keepdim=True)
            - 2 * z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(-1)
        )
        idx = d.argmin(-1)
        z_q = self.codebook(idx)

        commit_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = codebook_loss + self.beta * commit_loss

        # straight-through: gradient flows to z, codebook updated by codebook_loss
        z_q_st = z + (z_q - z).detach()

        return {
            "z_used": z_q_st,                              # consumed by FDM (with ST grad)
            "z_eval": z_q,                                 # deterministic, for probing
            "loss": loss,
            "aux": {"codes": idx.detach()},
        }


class GaussianBottleneck(nn.Module):
    """
    KL-regularized Gaussian bottleneck with reparameterization + free bits.

    The free-bits trick (Kingma et al., Improved Variational Inference, 2016)
    clamps per-dim KL at a minimum, preventing posterior collapse — the
    primary failure mode of continuous LAMs.

    `beta` is set externally during training (warmed up from 0 over the
    first kl_warmup_steps).
    """

    branch_name = "gaussian"

    def __init__(self, dim: int = 64, free_bits: float = 0.5):
        super().__init__()
        self.dim = dim
        self.free_bits = free_bits
        # head produces mu and log_var concatenated
        self.head = nn.Linear(dim, 2 * dim)
        # beta is updated externally per training step (KL warmup); register as
        # buffer so it persists in checkpoints
        self.register_buffer("beta", torch.tensor(0.0))

    def forward(self, z: torch.Tensor) -> dict[str, Any]:
        params = self.head(z)
        mu, log_var = params.chunk(2, dim=-1)
        log_var = log_var.clamp(-10.0, 10.0)  # numerical stability
        std = torch.exp(0.5 * log_var)

        if self.training:
            eps = torch.randn_like(std)
            z_sampled = mu + std * eps
        else:
            z_sampled = mu

        # standard KL(N(mu, sigma^2) || N(0, I)), per-dim
        kl_per_dim = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1.0)
        # free bits: don't penalize KL below the floor
        kl_clamped = torch.clamp(kl_per_dim, min=self.free_bits)
        kl_loss = kl_clamped.sum(dim=-1).mean()
        loss = self.beta * kl_loss

        return {
            "z_used": z_sampled,
            "z_eval": mu,
            "loss": loss,
            "aux": {
                "kl_raw": kl_per_dim.sum(dim=-1).mean().detach(),
                "kl_eff": kl_loss.detach(),
                "active_dims": (kl_per_dim.detach().mean(0) > 0.05).sum(),
                "mu_norm": mu.norm(dim=-1).mean().detach(),
                "logvar_mean": log_var.mean().detach(),
                "beta": self.beta.detach(),
            },
        }


def make_bottleneck(kind: str, dim: int, **kwargs) -> nn.Module:
    if kind == "vq":
        return VQBottleneck(dim=dim, **kwargs)
    if kind == "gaussian":
        return GaussianBottleneck(dim=dim, **kwargs)
    raise ValueError(f"unknown bottleneck: {kind}")


# =============================================================================
# LAM
# =============================================================================

class LAM(nn.Module):
    """LAPO-style LAM in feature space. ~3M trainable params (excl. encoder)."""

    def __init__(
        self,
        encoder: nn.Module,
        bottleneck: nn.Module,
        feat_dim: int = 384,
        latent_dim: int = 64,
        idm_hidden: int = 256,
        fdm_hidden: int = 512,
    ):
        super().__init__()
        self.encoder = encoder  # frozen
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.idm = MLP(feat_dim * 2, latent_dim, hidden=idm_hidden, depth=3)
        self.bottleneck = bottleneck
        self.fdm = MLP(feat_dim + latent_dim, feat_dim, hidden=fdm_hidden, depth=3)

    @torch.no_grad()
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) -> features (B, T, feat_dim)"""
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        feats = self.encoder(flat)
        return feats.reshape(B, T, -1)

    def forward(self, frames: torch.Tensor) -> dict[str, Any]:
        feats = self.extract_features(frames)
        feat_t = feats[:, :-1]
        feat_tp1 = feats[:, 1:]
        B, Tm1, D = feat_t.shape

        f_t = feat_t.reshape(B * Tm1, D)
        f_tp1 = feat_tp1.reshape(B * Tm1, D)

        z_pre = self.idm(torch.cat([f_t, f_tp1], dim=-1))
        bn = self.bottleneck(z_pre)

        f_pred = self.fdm(torch.cat([f_t, bn["z_used"]], dim=-1))
        recon_loss = F.mse_loss(f_pred, f_tp1)
        loss = recon_loss + bn["loss"]

        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "bottleneck_loss": bn["loss"].detach(),
            # latents — both branches expose pre-bottleneck and deterministic eval latent
            "z_pre": z_pre.detach(),                      # IDM output, before bottleneck
            "z_used": bn["z_used"].detach(),              # what FDM consumed
            "z_eval": bn["z_eval"].detach(),              # deterministic — use for probes
            # features
            "feat_t": f_t.detach(),
            "feat_tp1": f_tp1.detach(),
            "feat_pred": f_pred.detach(),
            # branch-specific diagnostics
            "aux": bn["aux"],
        }


# =============================================================================
# Dead-zone diagnostic (continuous branch only)
# =============================================================================

@torch.no_grad()
def measure_dead_zone(
    model: LAM,
    feat_t: torch.Tensor,
    n_samples: int = 256,
    threshold_quantile: float = 0.95,
) -> dict[str, float]:
    """
    For a continuous LAM: sample z from the prior N(0, I), run FDM(f_t, z),
    and measure the distribution of FDM outputs. Dead zones manifest as
    high-variance, low-coverage regions — a known failure mode that scaling
    is supposed to fix.

    Metric: fraction of (f_t, z_prior) pairs whose FDM output has prediction
    error in the top quantile of *training-distribution* errors. Lower is
    better.

    feat_t: (N, feat_dim) anchor frames from real data.
    """
    if not isinstance(model.bottleneck, GaussianBottleneck):
        raise ValueError("dead-zone metric only defined for GaussianBottleneck")

    N, D = feat_t.shape
    device = feat_t.device

    # sample z from prior
    z_prior = torch.randn(N * n_samples, model.latent_dim, device=device)
    f_t_rep = feat_t.unsqueeze(1).expand(N, n_samples, D).reshape(-1, D)
    f_pred = model.fdm(torch.cat([f_t_rep, z_prior], dim=-1))

    # per-sample feature norm (proxy for "is this output in distribution?")
    out_norms = f_pred.norm(dim=-1)
    real_norms = feat_t.norm(dim=-1)
    threshold = torch.quantile(real_norms, threshold_quantile)

    out_of_dist_frac = float((out_norms > threshold).float().mean())
    pred_diversity = float(f_pred.std(dim=0).mean())

    return {
        "dead_zone_ood_fraction": out_of_dist_frac,
        "prior_pred_diversity": pred_diversity,
        "real_norm_p95": float(threshold),
    }
