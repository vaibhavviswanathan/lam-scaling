"""
Fit a Chinchilla-style scaling law L(D) = L_inf + A * D^(-alpha) per branch
to the held-out reconstruction MSE produced by scaling.py.

Usage:
    python scripts/fit_powerlaw.py runs/scaling.json
    python scripts/fit_powerlaw.py runs/scaling.json --md   # markdown snippet

The fit is the simplest one consistent with finite-data noise floors. Uses
log-space NLLS via scipy.optimize.curve_fit; falls back to a 2-param fit
(L_inf=0) if scipy is missing or only 2 points are available.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence


def _curve_fit(hours: Sequence[float], losses: Sequence[float]) -> tuple[float, float, float]:
    """Returns (alpha, A, L_inf). Falls back to L_inf=0 if scipy unavailable
    or only 2 points are given."""
    if len(hours) < 2:
        raise ValueError("need at least 2 data points")
    if len(hours) < 3:
        # 2 points: fit L = A * D^(-alpha) (L_inf=0)
        d0, d1 = hours
        l0, l1 = losses
        alpha = (math.log(l0) - math.log(l1)) / (math.log(d1) - math.log(d0))
        A = l0 * d0**alpha
        return alpha, A, 0.0
    try:
        import numpy as np
        from scipy.optimize import curve_fit  # type: ignore
    except Exception:
        # 3-point fallback: log-linear fit on (log D, log L), L_inf = 0
        import numpy as np

        x = np.log(np.asarray(hours, dtype=float))
        y = np.log(np.asarray(losses, dtype=float))
        slope, intercept = np.polyfit(x, y, 1)
        return float(-slope), float(math.exp(intercept)), 0.0

    def model(D: "np.ndarray", L_inf: float, A: float, alpha: float) -> "np.ndarray":
        return L_inf + A * np.power(D, -alpha)

    p0 = (min(losses) * 0.5, max(losses), 0.5)
    bounds = ([0.0, 0.0, 0.0], [min(losses), 10 * max(losses), 5.0])
    popt, _ = curve_fit(
        model,
        np.asarray(hours, dtype=float),
        np.asarray(losses, dtype=float),
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )
    L_inf, A, alpha = popt
    return float(alpha), float(A), float(L_inf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scaling_json", type=str, default="runs/scaling.json", nargs="?")
    p.add_argument("--md", action="store_true", help="Emit a markdown snippet for proposal.md.")
    args = p.parse_args()

    rows = json.loads(Path(args.scaling_json).read_text())

    by_branch: dict[str, list[dict]] = {}
    for r in rows:
        by_branch.setdefault(r["branch"], []).append(r)

    fits: dict[str, dict] = {}
    for branch, brows in by_branch.items():
        brows = sorted(brows, key=lambda r: r["hours"])
        hours = [r["hours"] for r in brows]
        losses = [r["heldout_recon"] for r in brows]
        try:
            alpha, A, L_inf = _curve_fit(hours, losses)
            fits[branch] = {
                "alpha": alpha, "A": A, "L_inf": L_inf,
                "hours": hours, "heldout_recon": losses,
                "n_points": len(brows),
            }
        except Exception as e:
            fits[branch] = {"error": str(e), "hours": hours, "heldout_recon": losses}

    if args.md:
        lines = []
        for branch, f in fits.items():
            if "error" in f:
                lines.append(f"- **{branch}**: fit failed ({f['error']}); points {f['heldout_recon']}.")
                continue
            lines.append(
                f"- **{branch} branch:** α ≈ {f['alpha']:.3f}, "
                f"A ≈ {f['A']:.2f}, L∞ ≈ {f['L_inf']:.3f} "
                f"(over {f['n_points']} cells: hours {f['hours']}, "
                f"recon {[round(x, 3) for x in f['heldout_recon']]})."
            )
        print("\n".join(lines))
    else:
        print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
