"""
Hand-pose linear probe for a trained Latent Action Model.

Validates that the LAM's latent action z_t encodes something physically
meaningful — concretely, the *change* in 21-keypoint hand pose between
frame_t and frame_{t+1}. Run on multiple sources:

  - Egocentric-100K held-out  → in-distribution generalization
  - Ego4D / EPIC-KITCHENS clips → out-of-distribution transfer

The OOD R² is the headline number for the grant proposal: improvement at
scale on data the LAM has never seen is direct evidence that the
representation is transferable, not factory-memorized.

Setup:
    pip install mediapipe

Usage:
    # In-distribution (HF streaming, default Egocentric-100K)
    python hand_probe.py --ckpt lam_h100.pt --source hf

    # Out-of-distribution: download Ego4D / EPIC-KITCHENS clips locally
    python hand_probe.py --ckpt lam_h100.pt --source dir --path /data/ego4d_sample

Outputs JSON with the variance-weighted multivariate R², detection rate,
and per-keypoint summary statistics.
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import av
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import v2 as T

from eval import _load_lam


# ---------- Constants ----------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HAND_KEYPOINTS = 21
MAX_HANDS = 2
POSE_DIM = MAX_HANDS * HAND_KEYPOINTS * 2  # x, y per keypoint, two hands max


# ---------- MediaPipe wrapper ----------

class HandPoseExtractor:
    """Per-frame hand keypoints via MediaPipe Hands.

    Returns scale-invariant 2D keypoints (translated by wrist, scaled by
    wrist→index-MCP distance). Missing hands are zero-filled; presence
    indicated by a separate mask.

    NOTE: MediaPipe is not fork-safe, so call from main process only
    (DataLoader num_workers=0).
    """

    def __init__(self, min_confidence: float = 0.3):
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError("pip install mediapipe") from e
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )

    def process_clip(self, frames_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """frames_uint8: (T, H, W, 3) RGB uint8.
        Returns: poses (T, POSE_DIM) float32, mask (T,) float32."""
        T_total = frames_uint8.shape[0]
        poses = np.zeros((T_total, POSE_DIM), dtype=np.float32)
        mask = np.zeros(T_total, dtype=np.float32)
        for t, frame in enumerate(frames_uint8):
            res = self.hands.process(frame)
            if res.multi_hand_landmarks is None:
                continue
            mask[t] = 1.0
            for h, lm_set in enumerate(res.multi_hand_landmarks[:MAX_HANDS]):
                # 21 keypoints, x and y in image-relative coords [0, 1]
                kpts = np.array(
                    [[lm.x, lm.y] for lm in lm_set.landmark], dtype=np.float32
                )  # (21, 2)
                # scale-invariant: translate to wrist, normalize by palm size
                kpts -= kpts[0]
                scale = np.linalg.norm(kpts[5] - kpts[0]) + 1e-6
                kpts /= scale
                base = h * HAND_KEYPOINTS * 2
                poses[t, base : base + HAND_KEYPOINTS * 2] = kpts.flatten()
        return poses, mask

    def close(self):
        self.hands.close()


# ---------- Dataset that yields both normalized + raw frames ----------

def _decode_mp4(mp4_bytes: bytes) -> torch.Tensor:
    container = av.open(io.BytesIO(mp4_bytes))
    frames = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    finally:
        container.close()
    if not frames:
        raise ValueError("empty video")
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


class ProbeDataset(IterableDataset):
    """
    Yields tuples (clip_norm, clip_raw):
      - clip_norm: (T, 3, H, W) float32, ImageNet-normalized → goes into the LAM
      - clip_raw:  (T, H, W, 3) uint8 RGB     → goes into MediaPipe

    Sources:
      - source='hf':  stream from HuggingFace (default Egocentric-100K)
      - source='dir': iterate *.mp4 under a local directory (for Ego4D /
                      EPIC-KITCHENS samples downloaded locally)
    """

    def __init__(
        self,
        source: str = "hf",
        path: Optional[str] = None,
        repo_id: str = "builddotai/Egocentric-100K",
        clip_len: int = 8,
        stride: int = 8,
        image_size: int = 224,
        max_videos: int = 100,
        clips_per_video: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.source = source
        self.path = path
        self.repo_id = repo_id
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size
        self.max_videos = max_videos
        self.clips_per_video = clips_per_video
        self.seed = seed

        self.crop = T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
        ])
        self.norm = T.Compose([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _videos(self):
        if self.source == "hf":
            stream = load_dataset(self.repo_id, streaming=True)["train"].take(self.max_videos)
            for ex in stream:
                try:
                    yield _decode_mp4(ex["mp4"])
                except Exception:
                    continue
        elif self.source == "dir":
            assert self.path, "--path required when --source dir"
            files = sorted(Path(self.path).rglob("*.mp4"))[: self.max_videos]
            for f in files:
                try:
                    yield _decode_mp4(f.read_bytes())
                except Exception:
                    continue
        else:
            raise ValueError(f"unknown source: {self.source}")

    def __iter__(self):
        rng = random.Random(self.seed)
        for video in self._videos():
            T_total = video.shape[0]
            span = self.clip_len * self.stride
            if T_total < span:
                continue
            for _ in range(self.clips_per_video):
                start = rng.randint(0, T_total - span)
                idx = torch.arange(self.clip_len) * self.stride + start
                clip_uint8 = self.crop(video[idx])  # (T, C, H, W) uint8
                clip_norm = self.norm(clip_uint8)
                clip_raw = clip_uint8.permute(0, 2, 3, 1).numpy()  # (T, H, W, C) uint8
                yield clip_norm, clip_raw


def _collate(batch):
    norms = torch.stack([b[0] for b in batch])
    raws = np.stack([b[1] for b in batch])
    return norms, raws


# ---------- Probe ----------

def variance_weighted_r2(Z: torch.Tensor, target: torch.Tensor) -> Tuple[float, np.ndarray]:
    """Closed-form OLS multivariate R².
    Returns (variance-weighted scalar R², per-dim R² array)."""
    Z1 = torch.cat([Z.float(), torch.ones(len(Z), 1)], dim=1)
    sol = torch.linalg.lstsq(Z1, target.float()).solution
    pred = Z1 @ sol
    ss_res = (target - pred).pow(2).sum(0)
    ss_tot = (target - target.mean(0)).pow(2).sum(0).clamp_min(1e-12)
    per_dim = (1.0 - ss_res / ss_tot).numpy()
    weights = (ss_tot / ss_tot.sum()).numpy()
    return float(np.sum(per_dim * weights)), per_dim


@torch.no_grad()
def run_probe(
    ckpt: str,
    source: str = "hf",
    path: Optional[str] = None,
    repo_id: str = "builddotai/Egocentric-100K",
    max_videos: int = 100,
    clips_per_video: int = 2,
    num_clips: int = 200,
    batch_size: int = 4,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_lam(ckpt, device)

    ds = ProbeDataset(
        source=source,
        path=path,
        repo_id=repo_id,
        max_videos=max_videos,
        clips_per_video=clips_per_video,
    )
    # MediaPipe is not fork-safe; keep num_workers=0
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=_collate)

    extractor = HandPoseExtractor()
    Zs, deltas = [], []
    n_clips_seen = 0
    n_kept = 0
    n_total = 0

    try:
        for clip_norm, clip_raw in loader:
            if n_clips_seen >= num_clips:
                break
            B, T_, _, _, _ = clip_norm.shape
            n_clips_seen += B

            clip_norm = clip_norm.to(device, non_blocking=True)
            out = model(clip_norm)
            # z_eval: deterministic action representation (z_q for VQ, mu for Gaussian)
            # shape (B*(T-1), latent_dim) -> reshape per-clip
            z_all = out["z_eval"].reshape(B, T_ - 1, -1).cpu().numpy()

            for b in range(B):
                poses, mask = extractor.process_clip(clip_raw[b])  # (T, 84), (T,)
                for t in range(T_ - 1):
                    n_total += 1
                    if mask[t] > 0 and mask[t + 1] > 0:
                        Zs.append(z_all[b, t])
                        deltas.append(poses[t + 1] - poses[t])
                        n_kept += 1
    finally:
        extractor.close()

    if n_kept < 50:
        raise RuntimeError(
            f"Only {n_kept} valid transitions; need ≥50. "
            f"Detection rate {n_kept}/{n_total} = {n_kept/max(n_total,1):.1%}. "
            f"Try increasing --num_clips or --max_videos."
        )

    Z = torch.from_numpy(np.stack(Zs))
    D = torch.from_numpy(np.stack(deltas))

    overall_r2, per_dim_r2 = variance_weighted_r2(Z, D)

    metrics = {
        "ckpt": ckpt,
        "source": source,
        "path_or_repo": path if source == "dir" else repo_id,
        "n_transitions_kept": int(n_kept),
        "n_transitions_total": int(n_total),
        "hand_detection_rate": float(n_kept / max(n_total, 1)),
        "hand_pose_delta_R2": float(overall_r2),
        "per_dim_R2_mean": float(per_dim_r2.mean()),
        "per_dim_R2_median": float(np.median(per_dim_r2)),
        "per_dim_R2_max": float(per_dim_r2.max()),
        "n_dims_above_0p1": int((per_dim_r2 > 0.1).sum()),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--source", choices=["hf", "dir"], default="hf")
    p.add_argument("--path", default=None, help="local directory of mp4s when --source dir")
    p.add_argument("--repo_id", default="builddotai/Egocentric-100K")
    p.add_argument("--max_videos", type=int, default=100)
    p.add_argument("--clips_per_video", type=int, default=2)
    p.add_argument("--num_clips", type=int, default=200)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--out", default=None, help="optional path to dump metrics JSON")
    args = p.parse_args()

    metrics = run_probe(
        ckpt=args.ckpt,
        source=args.source,
        path=args.path,
        repo_id=args.repo_id,
        max_videos=args.max_videos,
        clips_per_video=args.clips_per_video,
        num_clips=args.num_clips,
        batch_size=args.bs,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
