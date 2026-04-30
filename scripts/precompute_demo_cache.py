"""
Precompute the demo cache: for each of the 6 sweep checkpoints, encode N
held-out transitions and save (f_t, f_{t+1}, z_eval, motion magnitude,
frame_t.jpg, frame_{t+1}.jpg) to disk so the Gradio app can load
instantly without holding DINOv2 in memory.

Layout under --out:

    runs/demo_cache/
      frames/
        000_t.jpg, 000_tp1.jpg, ...   # shared across checkpoints (same transitions)
      vq_h1.pt
      vq_h10.pt
      vq_h100.pt
      gaussian_h1.pt
      gaussian_h10.pt
      gaussian_h100.pt
      meta.json    # per-transition source video + frame indices

Each .pt is a dict with keys:
    f_t        (N, 384)  float32  DINOv2 feature
    f_tp1      (N, 384)  float32
    z_eval     (N, 64)   float32  the deterministic latent action
    motion     (N,)      float32  ||f_tp1 - f_t||
    branch     str
    hours      float
    aux        dict (codes for VQ; active_dims_mask for Gaussian)

Usage:
    python scripts/precompute_demo_cache.py \
        --ckpt_dir runs/sweep \
        --data_path /home/vai/data/ego100k_heldout \
        --n_transitions 500 \
        --out runs/demo_cache
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import LocalClipDataset, _decode_mp4
from eval import _load_lam


def _denorm(clip_norm: torch.Tensor) -> np.ndarray:
    """Reverse the ImageNet normalization for visualization. Returns HWC uint8."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    arr = (clip_norm.cpu().float() * std + mean).clamp(0, 1) * 255
    return arr.byte().permute(0, 2, 3, 1).numpy()  # (T, H, W, 3)


def collect_transitions(
    data_path: str,
    n_transitions: int,
    out_dir: Path,
    image_size: int = 224,
    seed: int = 999,
    clip_len: int = 8,
    stride: int = 8,
    clips_per_video: int = 5,
):
    """Sample N held-out transitions and save raw frame_t + frame_{t+1} thumbnails.

    Returns the normalized clip tensors for each transition (used downstream by
    every checkpoint), along with metadata.
    """
    out_frames = out_dir / "frames"
    out_frames.mkdir(parents=True, exist_ok=True)

    ds = LocalClipDataset(
        path=data_path,
        clip_len=clip_len,
        stride=stride,
        image_size=image_size,
        max_videos=200,
        clips_per_video=clips_per_video,
        seed=seed,
        loop=True,
    )
    loader = DataLoader(ds, batch_size=4, num_workers=0)

    # We collect (frame_t_norm, frame_tp1_norm) tensors and corresponding
    # uint8 thumbnails. Each "clip" in the dataset has clip_len frames, so
    # produces clip_len-1 transitions; we use them all.
    clips_norm: list[torch.Tensor] = []   # (T, 3, H, W) per clip
    sources: list[dict] = []
    transitions = 0

    for batch in loader:
        # batch: (B, T, 3, H, W) normalized
        for clip in batch:
            if transitions >= n_transitions:
                break
            clips_norm.append(clip.clone())
            # Save frame_t and frame_{t+1} thumbnails for each transition in
            # the clip (clip_len - 1 transitions per clip).
            uint8 = _denorm(clip)
            for t in range(clip.shape[0] - 1):
                idx = transitions
                Image.fromarray(uint8[t]).save(
                    out_frames / f"{idx:05d}_t.jpg", quality=80
                )
                Image.fromarray(uint8[t + 1]).save(
                    out_frames / f"{idx:05d}_tp1.jpg", quality=80
                )
                sources.append({
                    "transition_id": idx,
                    "clip_id": len(clips_norm) - 1,
                    "frame_in_clip": t,
                })
                transitions += 1
                if transitions >= n_transitions:
                    break
        if transitions >= n_transitions:
            break

    if transitions < n_transitions:
        print(f"  warning: only collected {transitions}/{n_transitions} transitions "
              f"(ran out of held-out clips)")

    print(f"  wrote {transitions} pairs of thumbnails to {out_frames}")
    (out_dir / "meta.json").write_text(json.dumps({
        "n_transitions": transitions,
        "image_size": image_size,
        "clip_len": clip_len,
        "stride": stride,
        "data_path": data_path,
        "seed": seed,
        "sources": sources,
    }, indent=2))
    return clips_norm, transitions


@torch.no_grad()
def encode_with_checkpoint(
    ckpt_path: Path,
    clips_norm: list[torch.Tensor],
    n_transitions: int,
    device: str,
    out_pt: Path,
):
    """Run all clips through a single checkpoint, collect per-transition tensors."""
    print(f"  loading {ckpt_path.name}")
    model = _load_lam(str(ckpt_path), device)
    model.eval()
    branch = model.bottleneck.branch_name

    # Process clips in batches of 4 to keep memory bounded
    f_t_list: list[torch.Tensor] = []
    f_tp1_list: list[torch.Tensor] = []
    z_eval_list: list[torch.Tensor] = []
    aux_list: list[torch.Tensor] = []

    bs = 4
    transitions_seen = 0
    for i in range(0, len(clips_norm), bs):
        batch = torch.stack(clips_norm[i : i + bs]).to(device, non_blocking=True)
        out = model(batch)
        # out["feat_t"] etc are (B*(T-1), D)
        f_t_list.append(out["feat_t"].cpu())
        f_tp1_list.append(out["feat_tp1"].cpu())
        z_eval_list.append(out["z_eval"].cpu())
        if branch == "vq":
            aux_list.append(out["aux"]["codes"].cpu())
        # else: gaussian aux is per-batch scalars, skip per-transition

    f_t = torch.cat(f_t_list)[:n_transitions].float()
    f_tp1 = torch.cat(f_tp1_list)[:n_transitions].float()
    z_eval = torch.cat(z_eval_list)[:n_transitions].float()
    motion = (f_tp1 - f_t).norm(dim=-1).float()

    rec = {
        "f_t": f_t,
        "f_tp1": f_tp1,
        "z_eval": z_eval,
        "motion": motion,
        "branch": branch,
    }
    if aux_list:
        rec["codes"] = torch.cat(aux_list)[:n_transitions].long()

    # Also record the FDM weights so the app can re-run interpolation. Save
    # the full state_dict (small): idm + bottleneck + fdm. The app will
    # reload via _load_lam() — so we just need the path. Skip saving weights
    # here.

    torch.save(rec, out_pt)
    print(
        f"  -> {out_pt.name}  n_transitions={len(f_t)}  "
        f"motion p50={motion.median().item():.2f}  "
        f"motion p95={motion.quantile(0.95).item():.2f}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default="runs/sweep")
    p.add_argument("--data_path", default="/home/vai/data/ego100k_heldout")
    p.add_argument("--n_transitions", type=int, default=500)
    p.add_argument("--out", default="runs/demo_cache")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Collect transitions + frame thumbnails (shared across checkpoints).
    print(f"[step 1] collecting {args.n_transitions} held-out transitions ...")
    clips_norm, n_real = collect_transitions(
        data_path=args.data_path,
        n_transitions=args.n_transitions,
        out_dir=out_dir,
    )

    # 2) For each checkpoint, encode the same clips and save tensors.
    ckpt_paths = sorted(Path(args.ckpt_dir).glob("lam_*.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"no checkpoints under {args.ckpt_dir}")
    print(f"[step 2] encoding {len(ckpt_paths)} checkpoints x {n_real} transitions ...")

    for ckpt in ckpt_paths:
        # name like "lam_vq_h1_s1500.pt" -> "vq_h1.pt"
        parts = ckpt.stem.split("_")  # ["lam", "vq", "h1", "s1500"]
        if len(parts) >= 3:
            tag = f"{parts[1]}_{parts[2]}"
        else:
            tag = ckpt.stem
        out_pt = out_dir / f"{tag}.pt"
        encode_with_checkpoint(
            ckpt_path=ckpt,
            clips_norm=clips_norm,
            n_transitions=n_real,
            device=device,
            out_pt=out_pt,
        )

    print(f"\nDone. Cache at {out_dir}")
    print(f"  meta.json + frames/ + {len(ckpt_paths)} per-checkpoint .pt files")


if __name__ == "__main__":
    main()
