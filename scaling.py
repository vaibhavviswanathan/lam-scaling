"""
Data-scaling sweep across both bottlenecks. Produces the crossover figure
that goes in the proposal.

Usage:
    # both branches (the proposal artifact)
    python scaling.py --branches vq gaussian --hours 1 10 100 --steps 1500 4000 10000

    # single branch
    python scaling.py --branches vq --hours 1 10 100 --steps 1500 4000 10000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from eval import evaluate
from train import train


def run_sweep(
    branches: list[str],
    hours_grid: list[float],
    steps_grid: list[int],
    out_dir: str = "runs",
    bs: int = 16,
    source: str = "hf",
    data_path: str | None = None,
    eval_data_path: str | None = None,
    eval_hours: float = 1.0,
    num_workers: int = 6,
    log_every: int = 25,
    vq_ema: bool = False,
    vq_num_codes: int = 64,
):
    """Run the data-scaling sweep across one or both branches.

    `data_path` is the training cache (used when source='dir').
    `eval_data_path` is the held-out cache; falls back to `data_path` if None,
    in which case eval is on the same videos with a different seed.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    assert len(hours_grid) == len(steps_grid)
    eval_path = eval_data_path or data_path

    for branch in branches:
        for hours, steps in zip(hours_grid, steps_grid):
            tag = f"{branch}_h{hours:g}_s{steps}"
            ckpt = out / f"lam_{tag}.pt"
            print(f"\n========== {tag} ==========")
            train_kwargs = dict(
                bottleneck=branch,
                hours=hours,
                steps=steps,
                batch_size=bs,
                save_path=str(ckpt),
                source=source,
                data_path=data_path,
                num_workers=num_workers,
                log_every=log_every,
            )
            if branch == "vq":
                train_kwargs["num_codes"] = vq_num_codes
                train_kwargs["vq_ema"] = vq_ema
            train_metrics = train(**train_kwargs)
            eval_metrics = evaluate(
                ckpt=str(ckpt),
                hours=eval_hours,
                batch_size=bs,
                num_batches=50,
                dead_zone=(branch == "gaussian"),
                source=source,
                data_path=eval_path,
            )
            row = {
                "branch": branch,
                "hours": hours,
                "steps": steps,
                "train_final_recon": train_metrics["final_recon"],
                "heldout_recon": eval_metrics["heldout_recon_mse"],
                "motion_probe_r2": eval_metrics["motion_probe_r2"],
            }
            if branch == "vq":
                row.update({
                    "codebook_perplexity": eval_metrics["codebook_perplexity"],
                    "codes_used": eval_metrics["codes_used"],
                })
            else:
                row.update({
                    "kl_raw": eval_metrics.get("gauss_kl_raw"),
                    "active_dims": eval_metrics.get("gauss_active_dims"),
                    "dead_zone_ood_fraction": eval_metrics.get("dead_zone_dead_zone_ood_fraction"),
                })
            rows.append(row)
            (out / "scaling.json").write_text(json.dumps(rows, indent=2))

    # plot — crossover figure if both branches present
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"vq": "#1f77b4", "gaussian": "#d62728"}
    labels = {"vq": "Discrete (VQ)", "gaussian": "Continuous (Gaussian)"}

    for branch in branches:
        b_rows = [r for r in rows if r["branch"] == branch]
        if not b_rows:
            continue
        hrs = [r["hours"] for r in b_rows]
        loss = [r["heldout_recon"] for r in b_rows]
        r2 = [r["motion_probe_r2"] for r in b_rows]
        axes[0].loglog(hrs, loss, "o-", color=colors[branch], label=labels[branch])
        axes[1].semilogx(hrs, r2, "o-", color=colors[branch], label=labels[branch])

    axes[0].set_xlabel("Hours of training data")
    axes[0].set_ylabel("Held-out feature MSE")
    axes[0].set_title("Data scaling: held-out reconstruction")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Hours of training data")
    axes[1].set_ylabel("Motion-probe R² (use hand_probe.py for headline)")
    axes[1].set_title("Latent-action probe quality")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out / "scaling.png", dpi=150)
    print(f"\nWrote {out/'scaling.json'} and {out/'scaling.png'}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--branches", nargs="+", default=["vq", "gaussian"],
                   choices=["vq", "gaussian"])
    p.add_argument("--hours", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    p.add_argument("--steps", type=int, nargs="+", default=[1500, 4000, 10000])
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--source", choices=["hf", "dir"], default="hf")
    p.add_argument("--data_path", type=str, default=None,
                   help="training mp4 cache; required when --source dir")
    p.add_argument("--eval_data_path", type=str, default=None,
                   help="held-out mp4 cache for eval; falls back to --data_path")
    p.add_argument("--eval_hours", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--vq_ema", action="store_true",
                   help="Use EMA codebook for the VQ branch (recommended).")
    p.add_argument("--vq_num_codes", type=int, default=64)
    args = p.parse_args()
    run_sweep(
        branches=args.branches,
        hours_grid=args.hours,
        steps_grid=args.steps,
        out_dir=args.out_dir,
        bs=args.bs,
        source=args.source,
        data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        eval_hours=args.eval_hours,
        num_workers=args.num_workers,
        log_every=args.log_every,
        vq_ema=args.vq_ema,
        vq_num_codes=args.vq_num_codes,
    )
