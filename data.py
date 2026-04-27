"""
Streaming clip loader for Egocentric-100K (and any other HF webdataset/parquet
of `mp4` blobs that follows the same convention).

Yields (T, C, H, W) float32 ImageNet-normalized clips. The DataLoader stacks
them into (B, T, C, H, W) for the LAM forward pass.

Design choices:
  - IterableDataset so we can stream from HuggingFace without downloading the
    full corpus.
  - Worker-aware sharding: with num_workers>0 each worker advances a different
    slice of the stream so we don't yield duplicates.
  - Per-clip ImageNet normalization here, so train/eval don't have to.
"""
from __future__ import annotations

import io
import random
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import v2 as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _decode_mp4(mp4_bytes: bytes, resize_short: int | None = None) -> torch.Tensor:
    """Decode mp4 bytes -> (T, C, H, W) uint8 RGB tensor.

    If `resize_short` is given, each frame is rescaled in-decoder so its
    shorter side matches `resize_short`; this dramatically reduces peak RAM
    when decoding long clips on a host with many DataLoader workers.
    """
    import av  # local import: keeps module importable without av

    container = av.open(io.BytesIO(mp4_bytes))
    stream = container.streams.video[0]
    if resize_short is not None and stream.width > 0 and stream.height > 0:
        if stream.width < stream.height:
            new_w = resize_short
            new_h = int(round(stream.height * resize_short / stream.width))
        else:
            new_h = resize_short
            new_w = int(round(stream.width * resize_short / stream.height))
    else:
        new_w = new_h = None

    frames = []
    try:
        for frame in container.decode(video=0):
            if new_w is not None:
                frame = frame.reformat(width=new_w, height=new_h, format="rgb24")
            arr = frame.to_ndarray(format="rgb24")
            frames.append(arr)
    finally:
        container.close()
    if not frames:
        raise ValueError("empty video")
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


class EgocentricClipDataset(IterableDataset):
    """Stream clips from an HF dataset of mp4 blobs.

    Each sample yielded is a (clip_len, 3, image_size, image_size) float32
    tensor, ImageNet-normalized.
    """

    def __init__(
        self,
        clip_len: int = 8,
        stride: int = 8,
        image_size: int = 224,
        max_videos: int = 200,
        clips_per_video: int = 4,
        seed: int = 0,
        repo_id: str = "builddotai/Egocentric-100K",
        split: str = "train",
        mp4_field: str = "mp4",
    ):
        super().__init__()
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size
        self.max_videos = max(1, int(max_videos))
        self.clips_per_video = clips_per_video
        self.seed = seed
        self.repo_id = repo_id
        self.split = split
        self.mp4_field = mp4_field

        self.crop = T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
        ])
        self.norm = T.Compose([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ---- video stream ----------------------------------------------------

    def _videos(self, worker_id: int, num_workers: int) -> Iterator[torch.Tensor]:
        from datasets import load_dataset  # local import

        stream = load_dataset(self.repo_id, streaming=True)[self.split]
        stream = stream.take(self.max_videos)

        for i, ex in enumerate(stream):
            # shard across workers: each worker takes every Nth example
            if i % num_workers != worker_id:
                continue
            blob = ex.get(self.mp4_field)
            if blob is None:
                # some HF webdatasets nest the bytes under {"bytes": ...}
                for v in ex.values():
                    if isinstance(v, dict) and "bytes" in v:
                        blob = v["bytes"]
                        break
            if blob is None:
                continue
            try:
                yield _decode_mp4(blob, resize_short=self.image_size)
            except Exception:
                continue

    # ---- iterator --------------------------------------------------------

    def __iter__(self) -> Iterator[torch.Tensor]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        rng = random.Random(self.seed + 10_000 * worker_id)

        span = self.clip_len * self.stride

        for video in self._videos(worker_id, num_workers):
            T_total = video.shape[0]
            if T_total < span:
                continue
            for _ in range(self.clips_per_video):
                start = rng.randint(0, T_total - span)
                idx = torch.arange(self.clip_len) * self.stride + start
                clip_uint8 = self.crop(video[idx])  # (T, C, H, W) uint8
                yield self.norm(clip_uint8)         # (T, C, H, W) float32


# ---- Local directory source -----------------------------------------------


class LocalClipDataset(IterableDataset):
    """Iterate over `*.mp4` files in a local directory.

    Same output contract as EgocentricClipDataset — a (T, C, H, W) float32
    ImageNet-normalized clip per yield. Useful when the HF dataset is gated /
    unavailable, or when probing OOD video.
    """

    def __init__(
        self,
        path: str,
        clip_len: int = 8,
        stride: int = 8,
        image_size: int = 224,
        max_videos: int = 200,
        clips_per_video: int = 4,
        seed: int = 0,
        loop: bool = False,
    ):
        super().__init__()
        from pathlib import Path

        self.files = sorted(Path(path).rglob("*.mp4"))[: max(1, int(max_videos))]
        if not self.files:
            raise FileNotFoundError(f"No *.mp4 under {path}")
        self.clip_len = clip_len
        self.stride = stride
        self.image_size = image_size
        self.clips_per_video = clips_per_video
        self.seed = seed
        self.loop = loop

        self.crop = T.Compose([
            T.Resize(image_size, antialias=True),
            T.CenterCrop(image_size),
        ])
        self.norm = T.Compose([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __iter__(self) -> Iterator[torch.Tensor]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        rng = random.Random(self.seed + 10_000 * worker_id)
        span = self.clip_len * self.stride

        epoch = 0
        while True:
            for i, f in enumerate(self.files):
                if i % num_workers != worker_id:
                    continue
                try:
                    video = _decode_mp4(f.read_bytes(), resize_short=self.image_size)
                except Exception:
                    continue
                T_total = video.shape[0]
                if T_total < span:
                    del video
                    continue
                # extract clips from this video, then drop it before reading
                # the next file — keeps peak RAM bounded under high worker counts
                clips = []
                for _ in range(self.clips_per_video):
                    start = rng.randint(0, T_total - span)
                    idx = torch.arange(self.clip_len) * self.stride + start
                    clips.append(self.crop(video[idx]).clone())
                del video
                for clip_uint8 in clips:
                    yield self.norm(clip_uint8)
            epoch += 1
            if not self.loop:
                return


# ---- Synthetic fallback for offline smoke tests ----------------------------


class SyntheticClipDataset(IterableDataset):
    """Random-noise clips with the same shape contract as EgocentricClipDataset.

    Lets verification tests run with no network or HF dataset access. The
    distribution is meaningless — only shapes and dtypes match.
    """

    def __init__(
        self,
        clip_len: int = 8,
        image_size: int = 224,
        n_clips: int = 64,
        seed: int = 0,
    ):
        super().__init__()
        self.clip_len = clip_len
        self.image_size = image_size
        self.n_clips = n_clips
        self.seed = seed

    def __iter__(self) -> Iterator[torch.Tensor]:
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        gen = torch.Generator().manual_seed(self.seed + worker_id)
        for _ in range(self.n_clips):
            yield torch.randn(
                self.clip_len, 3, self.image_size, self.image_size,
                generator=gen,
            )
