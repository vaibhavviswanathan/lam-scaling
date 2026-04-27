"""
One-shot prefetch: pull N videos from the (gated) Egocentric-100K HF stream
and write the raw mp4 bytes to a local directory. After that, train.py can
use --source dir --data_path <that dir> for fast, GPU-bound training.

Usage:
    python scripts/cache_dataset.py --out /home/vai/data/ego100k_cache --n 200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--repo_id", type=str, default="builddotai/Egocentric-100K")
    p.add_argument("--split", type=str, default="train")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("ego_*.mp4"))
    start_idx = len(existing)
    if start_idx >= args.n:
        print(f"Already have {start_idx} files in {out}, target {args.n} — skipping.")
        return
    print(f"Caching to {out} (have {start_idx}, target {args.n}).")

    stream = load_dataset(args.repo_id, streaming=True)[args.split]
    stream = stream.skip(start_idx).take(args.n - start_idx)

    t0 = time.time()
    written = start_idx
    bytes_total = 0
    for ex in stream:
        blob = ex.get("mp4")
        if blob is None:
            continue
        path = out / f"ego_{written:06d}.mp4"
        path.write_bytes(blob)
        written += 1
        bytes_total += len(blob)
        if written % 5 == 0 or written == args.n:
            elapsed = time.time() - t0
            mb_s = bytes_total / 1e6 / max(elapsed, 1e-3)
            print(
                f"  [{written}/{args.n}]  "
                f"{bytes_total/1e6:.1f}MB  {elapsed:.0f}s  ({mb_s:.2f} MB/s)"
            )
    print(f"Done. {written} files in {out}.")


if __name__ == "__main__":
    main()
