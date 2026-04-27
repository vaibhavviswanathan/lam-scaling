"""
Evaluation probes for a trained LAM (either branch).

Probes:

1. Motion-magnitude linear probe (R²) — sanity check on z_eval.
2. Held-out feature-space reconstruction MSE.
3. Branch-specific:
     - Discrete (VQ): codebook perplexity, codes used.
     - Continuous (Gaussian): mean KL, active dim count, dead-zone metric.

Use `hand_probe.py` for the headline OOD validation.

Usage:
    python eval.py --ckpt lam_vq_h10.pt --hours 1.0
    python eval.py --ckpt lam_gauss_h10.pt --hours 1.0 --dead_zone
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import EgocentricClipDataset
from model import (
    LAM,
    GaussianBottleneck,
    VQBottleneck,
    load_dinov2_small,
    make_bottleneck,
    measure_dead_zone,
)


def _load_lam(ckpt_path: str, device: str) -> LAM:
    """Load a checkpoint of either branch. Reads bottleneck kind from config."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    bn_cfg = cfg["bottleneck"]
    kind = bn_cfg["kind"]

    encoder = load_dinov2_small().to(device)
    if kind == "vq":
        bn = make_bottleneck(
            "vq",
            dim=cfg["latent_dim"],
            num_codes=bn_cfg["num_codes"],
            beta=bn_cfg.get("beta", 0.25),
        )
    elif kind == "gaussian":
        bn = make_bottleneck(
            "gaussian",
            dim=cfg["latent_dim"],
            free_bits=bn_cfg.get("free_bits", 0.5),
        )
    else:
        raise ValueError(f"unknown bottleneck kind in checkpoint: {kind}")

    model = LAM(
        encoder=encoder,
        bottleneck=bn,
        feat_dim=cfg["feat_dim"],
        latent_dim=cfg["latent_dim"],
    ).to(device)
    model.idm.load_state_dict(ckpt["state_dict"]["idm"])
    model.bottleneck.load_state_dict(ckpt["state_dict"]["bottleneck"])
    model.fdm.load_state_dict(ckpt["state_dict"]["fdm"])
    model.eval()
    return model


@torch.no_grad()
def collect_outputs(model: LAM, loader: DataLoader, device: str, num_batches: int):
    """Returns (z_eval, codes_or_None, motion, recon, feat_t_sample)."""
    Zs, branch_aux, motions, recons = [], [], [], []
    feat_t_samples = []
    for i, clips in enumerate(loader):
        if i >= num_batches:
            break
        clips = clips.to(device, non_blocking=True)
        out = model(clips)
        Zs.append(out["z_eval"].cpu())
        if isinstance(model.bottleneck, VQBottleneck):
            branch_aux.append(out["aux"]["codes"].flatten().cpu())
        else:
            # gaussian: nothing to accumulate per-sample
            pass
        motion = (out["feat_tp1"] - out["feat_t"]).norm(dim=-1)
        motions.append(motion.cpu())
        per_sample = (out["feat_pred"] - out["feat_tp1"]).pow(2).mean(dim=-1)
        recons.append(per_sample.cpu())
        if i < 2:
            # save a small batch of feat_t for dead-zone diagnostic
            feat_t_samples.append(out["feat_t"][:128].cpu())
    return (
        torch.cat(Zs),
        torch.cat(branch_aux) if branch_aux else None,
        torch.cat(motions),
        torch.cat(recons),
        torch.cat(feat_t_samples) if feat_t_samples else None,
    )


def linear_probe_r2(Z: torch.Tensor, target: torch.Tensor) -> float:
    """Closed-form OLS R² from Z (with bias) onto target."""
    target = target.float().reshape(-1, 1)
    Z1 = torch.cat([Z.float(), torch.ones(len(Z), 1)], dim=1)
    sol = torch.linalg.lstsq(Z1, target).solution
    pred = Z1 @ sol
    ss_res = (target - pred).pow(2).sum()
    ss_tot = (target - target.mean()).pow(2).sum().clamp_min(1e-12)
    return float(1.0 - ss_res / ss_tot)


def codebook_perplexity(codes: torch.Tensor, num_codes: int) -> tuple[float, int]:
    counts = torch.bincount(codes.long(), minlength=num_codes).float()
    p = counts / counts.sum().clamp_min(1)
    nz = p[p > 0]
    perp = float(torch.exp(-(nz * nz.log()).sum()))
    return perp, int((counts > 0).sum())


def evaluate(
    ckpt: str,
    hours: float = 1.0,
    batch_size: int = 16,
    num_batches: int = 100,
    seed: int = 999,
    num_workers: int = 2,
    dead_zone: bool = False,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_lam(ckpt, device)
    branch = model.bottleneck.branch_name

    ds = EgocentricClipDataset(
        clip_len=8,
        stride=8,
        max_videos=max(1, int(hours * 20)),
        clips_per_video=2,
        seed=seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    Z, branch_aux, motion, recon, feat_t_sample = collect_outputs(
        model, loader, device, num_batches
    )

    metrics = {
        "ckpt": ckpt,
        "branch": branch,
        "eval_hours": hours,
        "n_transitions": int(len(Z)),
        "motion_probe_r2": linear_probe_r2(Z, motion),
        "heldout_recon_mse": float(recon.mean()),
    }

    if branch == "vq":
        perp, used = codebook_perplexity(branch_aux, model.bottleneck.num_codes)
        metrics.update({
            "codebook_perplexity": perp,
            "codes_used": used,
            "codes_total": model.bottleneck.num_codes,
        })
    else:  # gaussian
        # KL/active-dim already accumulated in training; report a fresh
        # measurement on held-out batch
        with torch.no_grad():
            sample = next(iter(loader)).to(device)
            out = model(sample)
            metrics.update({
                "gauss_kl_raw": float(out["aux"]["kl_raw"]),
                "gauss_active_dims": int(out["aux"]["active_dims"]),
                "gauss_mu_norm": float(out["aux"]["mu_norm"]),
                "gauss_logvar_mean": float(out["aux"]["logvar_mean"]),
            })
        if dead_zone and feat_t_sample is not None:
            dz = measure_dead_zone(model, feat_t_sample.to(device))
            metrics.update({f"dead_zone_{k}": v for k, v in dz.items()})

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--num_batches", type=int, default=100)
    p.add_argument("--dead_zone", action="store_true",
                   help="Run dead-zone diagnostic (continuous branch only)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    metrics = evaluate(
        ckpt=args.ckpt,
        hours=args.hours,
        batch_size=args.bs,
        num_batches=args.num_batches,
        dead_zone=args.dead_zone,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
