"""
Smoke tests: catch shape/dtype/training-loop regressions before kicking off a
multi-hour run. These tests do NOT exercise the HF stream — they use the
SyntheticClipDataset fallback so they're self-contained and fast.

Run:
    pytest -q tests/test_smoke.py
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data import SyntheticClipDataset
from model import LAM, make_bottleneck


# A tiny stand-in for DINOv2 so we can test the wiring end-to-end without
# pulling 90MB from torch.hub.
class _FakeEncoder(torch.nn.Module):
    def __init__(self, feat_dim: int = 384):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj = torch.nn.Conv2d(3, feat_dim, kernel_size=14, stride=14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W) -> (N, feat_dim) global-average-pooled
        h = self.proj(x)
        return h.mean(dim=(-2, -1))


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_synthetic_dataset_shapes():
    ds = SyntheticClipDataset(clip_len=8, image_size=224, n_clips=4)
    items = list(ds)
    assert len(items) == 4
    for clip in items:
        assert clip.shape == (8, 3, 224, 224)
        assert clip.dtype == torch.float32


def test_dataloader_collates_to_batch():
    ds = SyntheticClipDataset(clip_len=8, image_size=224, n_clips=4)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch.shape == (2, 8, 3, 224, 224)
    assert batch.dtype == torch.float32


def _build_lam(branch: str, device: str) -> LAM:
    encoder = _FakeEncoder(feat_dim=384).to(device)
    bn_kwargs = {"vq": dict(num_codes=8, beta=0.25), "gaussian": dict(free_bits=0.5)}[branch]
    bn = make_bottleneck(branch, dim=64, **bn_kwargs)
    model = LAM(encoder=encoder, bottleneck=bn, feat_dim=384, latent_dim=64).to(device)
    return model


def test_vq_forward_one_step():
    device = _device()
    model = _build_lam("vq", device)
    clips = torch.randn(2, 8, 3, 224, 224, device=device)
    out = model(clips)
    # 2 batches * (8-1) transitions = 14
    assert out["z_eval"].shape == (14, 64)
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    assert out["aux"]["codes"].shape == (14,)


def test_gaussian_forward_one_step():
    device = _device()
    model = _build_lam("gaussian", device)
    # With training=True, the Gaussian branch samples — make sure no NaNs.
    model.train()
    # set beta>0 so KL contributes
    model.bottleneck.beta.fill_(0.1)
    clips = torch.randn(2, 8, 3, 224, 224, device=device)
    out = model(clips)
    assert out["z_eval"].shape == (14, 64)
    assert torch.isfinite(out["loss"])
    assert int(out["aux"]["active_dims"]) <= 64


def test_vq_optimizer_step_decreases_loss_on_repeated_batch():
    device = _device()
    torch.manual_seed(0)
    model = _build_lam("vq", device)
    trainable = list(model.idm.parameters()) + list(model.bottleneck.parameters()) + list(model.fdm.parameters())
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    clips = torch.randn(2, 8, 3, 224, 224, device=device)
    losses = []
    for _ in range(20):
        out = model(clips)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    # Overfitting a single fixed batch should drive total loss down meaningfully.
    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_gaussian_optimizer_step_decreases_loss_on_repeated_batch():
    device = _device()
    torch.manual_seed(0)
    model = _build_lam("gaussian", device)
    model.bottleneck.beta.fill_(0.0)  # no KL pressure: pure recon overfit
    trainable = list(model.idm.parameters()) + list(model.bottleneck.parameters()) + list(model.fdm.parameters())
    opt = torch.optim.AdamW(trainable, lr=1e-3)

    clips = torch.randn(2, 8, 3, 224, 224, device=device)
    losses = []
    for _ in range(20):
        out = model(clips)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_checkpoint_roundtrip(tmp_path):
    device = _device()
    model = _build_lam("vq", device)
    p = tmp_path / "lam.pt"
    torch.save(
        {
            "state_dict": {
                "idm": model.idm.state_dict(),
                "bottleneck": model.bottleneck.state_dict(),
                "fdm": model.fdm.state_dict(),
            },
            "config": {
                "feat_dim": 384,
                "latent_dim": 64,
                "bottleneck": {"kind": "vq", "num_codes": 8, "beta": 0.25},
            },
        },
        p,
    )
    blob = torch.load(p, map_location=device, weights_only=False)
    assert blob["config"]["bottleneck"]["kind"] == "vq"
    # Build a fresh model and confirm we can load weights.
    other = _build_lam("vq", device)
    other.idm.load_state_dict(blob["state_dict"]["idm"])
    other.bottleneck.load_state_dict(blob["state_dict"]["bottleneck"])
    other.fdm.load_state_dict(blob["state_dict"]["fdm"])
