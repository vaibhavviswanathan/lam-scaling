"""
Train a Latent Action Model on Egocentric-100K, with selectable bottleneck.

Usage:
    # Discrete (VQ) branch — same as before
    python train.py --bottleneck vq --hours 10 --steps 5000 --bs 16 --save lam_vq_h10.pt

    # Continuous (Gaussian) branch — adds KL warmup
    python train.py --bottleneck gaussian --hours 10 --steps 5000 --bs 16 --save lam_gauss_h10.pt

The two branches share architecture, optimizer, schedule, batch size, and
training compute per cell — only the bottleneck differs. This is the
controlled-comparison setup the proposal describes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import EgocentricClipDataset, LocalClipDataset
from model import LAM, load_dinov2_small, make_bottleneck


VIDEOS_PER_HOUR = 20  # ~3 min average per video


def train(
    bottleneck: str = "vq",
    hours: float = 10.0,
    steps: int = 5000,
    batch_size: int = 16,
    clip_len: int = 8,
    stride: int = 8,
    image_size: int = 224,
    lr: float = 3e-4,
    wd: float = 0.01,
    latent_dim: int = 64,
    # discrete-branch args
    num_codes: int = 64,
    vq_beta: float = 0.25,
    vq_ema: bool = False,
    vq_decay: float = 0.99,
    # continuous-branch args
    target_beta: float = 0.1,
    kl_warmup_steps: int = 500,
    free_bits: float = 0.5,
    # logging / output
    log_every: int = 50,
    save_path: str = "lam.pt",
    num_workers: int = 4,
    prefetch_factor: int = 2,
    seed: int = 0,
    source: str = "hf",
    data_path: str | None = None,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    torch.manual_seed(seed)

    max_videos = max(1, int(hours * VIDEOS_PER_HOUR))
    print(f"[{bottleneck}] source={source} up to {max_videos} videos (~{hours}h)")

    if source == "hf":
        ds = EgocentricClipDataset(
            clip_len=clip_len,
            stride=stride,
            image_size=image_size,
            max_videos=max_videos,
            clips_per_video=4,
            seed=seed,
        )
    elif source == "dir":
        if not data_path:
            raise ValueError("--data_path required when --source dir")
        # loop=True so a small directory still saturates `--steps` of training.
        ds = LocalClipDataset(
            path=data_path,
            clip_len=clip_len,
            stride=stride,
            image_size=image_size,
            max_videos=max_videos,
            clips_per_video=4,
            seed=seed,
            loop=True,
        )
    else:
        raise ValueError(f"unknown source: {source}")
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0 and prefetch_factor > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(ds, **loader_kwargs)

    encoder = load_dinov2_small().to(device)

    if bottleneck == "vq":
        bn = make_bottleneck(
            "vq",
            dim=latent_dim,
            num_codes=num_codes,
            beta=vq_beta,
            ema=vq_ema,
            decay=vq_decay,
        )
        bottleneck_config = {
            "kind": "vq",
            "num_codes": num_codes,
            "beta": vq_beta,
            "ema": vq_ema,
            "decay": vq_decay,
        }
    elif bottleneck == "gaussian":
        bn = make_bottleneck("gaussian", dim=latent_dim, free_bits=free_bits)
        bottleneck_config = {
            "kind": "gaussian",
            "free_bits": free_bits,
            "target_beta": target_beta,
            "kl_warmup_steps": kl_warmup_steps,
        }
    else:
        raise ValueError(f"unknown bottleneck: {bottleneck}")

    model = LAM(encoder=encoder, bottleneck=bn, latent_dim=latent_dim).to(device)

    trainable = (
        list(model.idm.parameters())
        + list(model.bottleneck.parameters())
        + list(model.fdm.parameters())
    )
    n_params = sum(p.numel() for p in trainable)
    print(f"[{bottleneck}] Trainable params: {n_params/1e6:.2f}M (encoder frozen)")

    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # Per-step CSV alongside the checkpoint, e.g. runs/lam_vq_h10.steps.csv
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    csv_path = save_p.with_suffix("") .parent / f"{save_p.stem}.steps.csv"
    csv_f = open(csv_path, "w", buffering=1)  # line-buffered
    csv_f.write("step,total,recon,bn,codes_used,active_dims,kl_raw,beta\n")

    history: list[dict] = []
    step = 0
    t0 = time.time()

    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            # KL warmup for the continuous branch
            if bottleneck == "gaussian":
                warmup_frac = min(1.0, step / max(1, kl_warmup_steps))
                model.bottleneck.beta.fill_(warmup_frac * target_beta)

            clips = batch.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                out = model(clips)
                loss = out["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            opt.step()
            sched.step()

            # CSV row every step (cheap; lets us plot smooth trajectories)
            rec_v = out["recon_loss"].item()
            bn_v = out["bottleneck_loss"].item()
            if bottleneck == "vq":
                codes_v = out["aux"]["codes"].unique().numel()
                csv_f.write(
                    f"{step},{loss.item():.6f},{rec_v:.6f},{bn_v:.6f},"
                    f"{codes_v},,,\n"
                )
            else:
                active_v = int(out["aux"]["active_dims"].item())
                kl_v = out["aux"]["kl_raw"].item()
                beta_v = float(out["aux"]["beta"].item())
                csv_f.write(
                    f"{step},{loss.item():.6f},{rec_v:.6f},{bn_v:.6f},,"
                    f"{active_v},{kl_v:.6f},{beta_v:.6f}\n"
                )

            if step % log_every == 0:
                elapsed = time.time() - t0
                rec = out["recon_loss"].item()
                bn_loss = out["bottleneck_loss"].item()
                aux_str = ""
                if bottleneck == "vq":
                    used = out["aux"]["codes"].unique().numel()
                    aux_str = f"  codes_used {used}/{num_codes}"
                else:  # gaussian
                    kl_raw = out["aux"]["kl_raw"].item()
                    active = int(out["aux"]["active_dims"].item())
                    aux_str = f"  kl_raw {kl_raw:.3f}  active_dims {active}/{latent_dim}"
                print(
                    f"[{bottleneck}] step {step:5d}  loss {loss.item():.4f}  "
                    f"recon {rec:.4f}  bn {bn_loss:.4f}{aux_str}  ({elapsed:.0f}s)"
                )
                history.append({
                    "step": step,
                    "loss": loss.item(),
                    "recon": rec,
                    "bottleneck_loss": bn_loss,
                })

            step += 1

    csv_f.close()

    final = {
        "bottleneck": bottleneck,
        "hours": hours,
        "steps": step,
        "final_loss": loss.item(),
        "final_recon": out["recon_loss"].item(),
        "final_bottleneck_loss": out["bottleneck_loss"].item(),
        "n_params_trainable": n_params,
        "history": history,
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                "idm": model.idm.state_dict(),
                "bottleneck": model.bottleneck.state_dict(),
                "fdm": model.fdm.state_dict(),
            },
            "config": {
                "feat_dim": model.feat_dim,
                "latent_dim": latent_dim,
                "bottleneck": bottleneck_config,
            },
            "metrics": final,
        },
        save_path,
    )
    print(f"Saved checkpoint -> {save_path}")
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bottleneck", choices=["vq", "gaussian"], default="vq")
    p.add_argument("--hours", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--latent_dim", type=int, default=64)
    # vq
    p.add_argument("--num_codes", type=int, default=64)
    p.add_argument("--vq_beta", type=float, default=0.25)
    p.add_argument("--vq_ema", action="store_true",
                   help="Use EMA codebook updates (recommended for scaling)")
    p.add_argument("--vq_decay", type=float, default=0.99)
    # gaussian
    p.add_argument("--target_beta", type=float, default=0.1)
    p.add_argument("--kl_warmup_steps", type=int, default=500)
    p.add_argument("--free_bits", type=float, default=0.5)
    # output
    p.add_argument("--save", type=str, default="lam.pt")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--source", choices=["hf", "dir"], default="hf",
                   help="hf: stream Egocentric-100K (gated). dir: local mp4 directory.")
    p.add_argument("--data_path", type=str, default=None,
                   help="path to local mp4 dir when --source dir")
    args = p.parse_args()

    result = train(
        bottleneck=args.bottleneck,
        hours=args.hours,
        steps=args.steps,
        batch_size=args.bs,
        lr=args.lr,
        latent_dim=args.latent_dim,
        num_codes=args.num_codes,
        vq_beta=args.vq_beta,
        vq_ema=args.vq_ema,
        vq_decay=args.vq_decay,
        target_beta=args.target_beta,
        kl_warmup_steps=args.kl_warmup_steps,
        free_bits=args.free_bits,
        save_path=args.save,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
        source=args.source,
        data_path=args.data_path,
        log_every=args.log_every,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
