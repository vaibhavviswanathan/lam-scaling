"""
FastAPI backend for the LAM latent action tour.

Endpoints:
    GET  /                                serve static index.html
    GET  /static/...                      static frontend assets (css, js)
    GET  /frames/<idx>_(t|tp1).jpg        the precomputed thumbnails
    GET  /api/checkpoints                 [{tag, label}, ...]
    GET  /api/cache/{tag}                 per-checkpoint scatter data
    POST /api/interpolate                 live FDM forward + retrieval

The client owns all UI state. The server is stateless except for cached
TagCache instances + lazily loaded model weights.

Run:
    python webdemo/server.py
    # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import _load_lam  # noqa: E402

CACHE_DIR = ROOT / "runs" / "demo_cache"
CKPT_DIR = ROOT / "runs" / "sweep"
FRAMES_DIR = CACHE_DIR / "frames"
WEB_DIR = ROOT / "webdemo"
STATIC_DIR = WEB_DIR / "static"

CHECKPOINT_TAGS = [
    ("vq_h1",         "Discrete (VQ-EMA), 1h",       "lam_vq_h1_s1500.pt"),
    ("vq_h10",        "Discrete (VQ-EMA), 10h",      "lam_vq_h10_s4000.pt"),
    ("vq_h100",       "Discrete (VQ-EMA), 100h",     "lam_vq_h100_s10000.pt"),
    ("gaussian_h1",   "Continuous (Gaussian), 1h",   "lam_gaussian_h1_s1500.pt"),
    ("gaussian_h10",  "Continuous (Gaussian), 10h",  "lam_gaussian_h10_s4000.pt"),
    ("gaussian_h100", "Continuous (Gaussian), 100h", "lam_gaussian_h100_s10000.pt"),
]
TAG_TO_CKPT = {t: c for t, _, c in CHECKPOINT_TAGS}
TAG_TO_LABEL = {t: l for t, l, _ in CHECKPOINT_TAGS}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _pad_2d(p: np.ndarray) -> np.ndarray:
    """Pad to 2 columns with zeros if a degenerate 1D projection comes back."""
    if p.shape[1] == 1:
        p = np.concatenate([p, np.zeros_like(p)], axis=1)
    return p


def _jitter(p: np.ndarray, rng) -> np.ndarray:
    """Tiny jitter so identical points (collapsed VQ codes) don't render
    exactly on top of each other and become unclickable."""
    return p + rng.normal(0, 0.005 * (np.std(p) + 1e-9), size=p.shape)


class TagCache:
    """Per-checkpoint cache: precomputed tensors + 2D projections of z_eval
    (PCA always; t-SNE and UMAP lazily, with a disk cache so subsequent
    server boots are instant).

    Model weights are loaded lazily — only when the first interpolation
    request arrives for a tag — so dropdown switches in the UI are cheap.
    """

    def __init__(self, tag: str):
        rec = torch.load(CACHE_DIR / f"{tag}.pt", map_location="cpu", weights_only=False)
        self.tag = tag
        self.f_t: torch.Tensor = rec["f_t"]
        self.f_tp1: torch.Tensor = rec["f_tp1"]
        self.z_eval: torch.Tensor = rec["z_eval"]
        self.motion: torch.Tensor = rec["motion"]
        self.branch: str = rec["branch"]
        self.codes: Optional[torch.Tensor] = rec.get("codes")

        z = self.z_eval.numpy().astype(np.float64)
        z_centered = z - z.mean(axis=0, keepdims=True)
        rng = np.random.default_rng(0)

        # PCA — always eager (fast).
        try:
            pca = PCA(n_components=min(2, z_centered.shape[1])).fit(z_centered)
            proj_pca = _pad_2d(pca.transform(z_centered))
        except Exception:
            proj_pca = np.zeros((len(z_centered), 2))
        self.proj_pca = _jitter(proj_pca, rng)

        # Disk cache for slower projections.
        self._proj_cache_path = CACHE_DIR / f"{tag}_projections.npz"
        self._proj_tsne: Optional[np.ndarray] = None
        self._proj_umap: Optional[np.ndarray] = None
        if self._proj_cache_path.exists():
            try:
                blob = np.load(self._proj_cache_path)
                if "tsne" in blob.files:
                    self._proj_tsne = blob["tsne"]
                if "umap" in blob.files:
                    self._proj_umap = blob["umap"]
            except Exception:
                pass

        self.motion_order = np.argsort(self.motion.numpy())
        self._model = None

    def _persist_projections(self) -> None:
        kwargs = {}
        if self._proj_tsne is not None:
            kwargs["tsne"] = self._proj_tsne
        if self._proj_umap is not None:
            kwargs["umap"] = self._proj_umap
        if kwargs:
            np.savez(self._proj_cache_path, **kwargs)

    def proj_tsne(self) -> np.ndarray:
        if self._proj_tsne is not None:
            return self._proj_tsne
        from sklearn.manifold import TSNE
        print(f"  computing t-SNE for {self.tag} ...", flush=True)
        z = self.z_eval.numpy().astype(np.float64)
        z = z - z.mean(axis=0, keepdims=True)
        # PCA-init t-SNE is faster + more reproducible than random-init.
        tsne = TSNE(
            n_components=2, init="pca", learning_rate="auto",
            perplexity=30, random_state=0,
        )
        proj = _pad_2d(tsne.fit_transform(z))
        self._proj_tsne = _jitter(proj, np.random.default_rng(1))
        self._persist_projections()
        return self._proj_tsne

    def proj_umap(self) -> np.ndarray:
        if self._proj_umap is not None:
            return self._proj_umap
        try:
            import umap as umap_lib
        except Exception:
            print(f"  umap-learn not installed; falling back to PCA for {self.tag}", flush=True)
            self._proj_umap = self.proj_pca.copy()
            return self._proj_umap
        print(f"  computing UMAP for {self.tag} ...", flush=True)
        z = self.z_eval.numpy().astype(np.float64)
        reducer = umap_lib.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=0,
        )
        proj = _pad_2d(reducer.fit_transform(z))
        self._proj_umap = _jitter(proj, np.random.default_rng(2))
        self._persist_projections()
        return self._proj_umap

    @property
    def model(self):
        if self._model is None:
            self._model = _load_lam(str(CKPT_DIR / TAG_TO_CKPT[self.tag]), DEVICE)
            self._model.eval()
        return self._model


_CACHES: dict[str, TagCache] = {}


def get_cache(tag: str) -> TagCache:
    if tag not in TAG_TO_CKPT:
        raise HTTPException(status_code=404, detail=f"unknown tag: {tag}")
    if tag not in _CACHES:
        _CACHES[tag] = TagCache(tag)
    return _CACHES[tag]


# ---- API ----

app = FastAPI(title="LAM latent action tour")


@app.get("/api/checkpoints")
def list_checkpoints():
    """Order matters; UI renders in this sequence."""
    return [{"tag": t, "label": l, "ckpt": c} for t, l, c in CHECKPOINT_TAGS]


@app.get("/api/cache/{tag}")
def cache_for_tag(tag: str):
    """Returns all three projections together so the client can toggle
    instantly. t-SNE and UMAP are computed lazily on first request and
    persisted to disk."""
    c = get_cache(tag)
    n = int(len(c.motion))
    pca = c.proj_pca
    tsne = c.proj_tsne()
    umap_p = c.proj_umap()
    return {
        "tag": tag,
        "label": TAG_TO_LABEL[tag],
        "branch": c.branch,
        "n": n,
        "transitions": [
            {
                "id": i,
                "motion": float(c.motion[i].item()),
                "pca":  [float(pca[i, 0]),  float(pca[i, 1])],
                "tsne": [float(tsne[i, 0]), float(tsne[i, 1])],
                "umap": [float(umap_p[i, 0]), float(umap_p[i, 1])],
                **({"code_id": int(c.codes[i].item())} if c.codes is not None else {}),
            }
            for i in range(n)
        ],
        "motion_order": c.motion_order.tolist(),
        "unique_codes": (
            int(c.codes.unique().numel()) if c.codes is not None else None
        ),
    }


class InterpBody(BaseModel):
    tag: str
    anchor_a: int
    anchor_b: int
    alpha: float


@app.post("/api/interpolate")
def interpolate(body: InterpBody):
    c = get_cache(body.tag)
    if not (0 <= body.anchor_a < len(c.f_t)) or not (0 <= body.anchor_b < len(c.f_t)):
        raise HTTPException(status_code=400, detail="anchor index out of range")
    if not (0.0 <= body.alpha <= 1.0):
        raise HTTPException(status_code=400, detail="alpha must be in [0, 1]")

    with torch.no_grad():
        z_a = c.z_eval[body.anchor_a].to(DEVICE)
        z_b = c.z_eval[body.anchor_b].to(DEVICE)
        z = (1.0 - body.alpha) * z_a + body.alpha * z_b
        f_t = c.f_t[body.anchor_a].to(DEVICE)
        pred = c.model.fdm(torch.cat([f_t, z], dim=-1))
        f_tp1_all = c.f_tp1.to(DEVICE)
        dists = (f_tp1_all - pred).pow(2).sum(dim=-1)
        nearest = int(dists.argmin().item())
        near_dist = float(dists[nearest].sqrt().item())
        f_t_all = c.f_t.to(DEVICE)
        far_dist = float((f_t_all - pred).pow(2).sum(dim=-1).min().sqrt().item())

    return {
        "nearest_idx": nearest,
        "alpha": body.alpha,
        "d_pred_to_nearest_tp1": near_dist,
        "d_pred_to_nearest_t": far_dist,
    }


# ---- static + index ----

# Frames mounted at /frames/<file>.jpg
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")
# Frontend assets at /static/...
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
