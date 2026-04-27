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
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    assert len(hours_grid) == len(steps_grid)

    for branch in branches:
        for hours, steps in zip(hours_grid, steps_grid):
            tag = f"{branch}_h{hours:g}_s{steps}"
            ckpt = out / f"lam_{tag}.pt"
            print(f"\n========== {tag} ==========")
            train_metrics = train(
                bottleneck=branch,
                hours=hours,
                steps=steps,
                batch_size=bs,
                save_path=str(ckpt),
            )
            eval_metrics = evaluate(
                ckpt=str(ckpt),
                hours=1.0,
                batch_size=bs,
                num_batches=50,
                dead_zone=(branch == "gaussian"),
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
    args = p.parse_args()
    run_sweep(args.branches, args.hours, args.steps, args.out_dir, bs=args.bs)
