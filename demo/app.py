"""
Local Gradio demo for the LAM crossover sweep.

Layout, top to bottom:

  - Branch dropdown.
  - Motion-percentile slider (the primary interaction). Drag low→high and
    watch the held-out transitions cycle from "stationary worker" to "active
    worker"; the scatter highlights where the current transition sits in the
    LAM's latent action space.
  - frame_t and frame_{t+1} of the current transition.
  - Scatter of z_eval (PCA-2D) colored by motion. Click any point as a
    shortcut to jump the slider to that transition.
  - Optional: A/B interpolation. Pin the current transition as A or B, then
    drag alpha; we feed the interpolated z through the FDM and retrieve the
    nearest real frame_{t+1}.

Usage:
    python demo/app.py
    # then open http://127.0.0.1:7860
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import _load_lam  # noqa: E402

CACHE_DIR = ROOT / "runs" / "demo_cache"
CKPT_DIR = ROOT / "runs" / "sweep"
FRAMES_DIR = CACHE_DIR / "frames"

CHECKPOINT_TAGS = [
    ("vq_h1",         "Discrete (VQ-EMA), 1h",       "lam_vq_h1_s1500.pt"),
    ("vq_h10",        "Discrete (VQ-EMA), 10h",      "lam_vq_h10_s4000.pt"),
    ("vq_h100",       "Discrete (VQ-EMA), 100h",     "lam_vq_h100_s10000.pt"),
    ("gaussian_h1",   "Continuous (Gaussian), 1h",   "lam_gaussian_h1_s1500.pt"),
    ("gaussian_h10",  "Continuous (Gaussian), 10h",  "lam_gaussian_h10_s4000.pt"),
    ("gaussian_h100", "Continuous (Gaussian), 100h", "lam_gaussian_h100_s10000.pt"),
]
TAG_TO_LABEL = {t: l for t, l, _ in CHECKPOINT_TAGS}
LABEL_TO_TAG = {l: t for t, l, _ in CHECKPOINT_TAGS}
TAG_TO_CKPT = {t: c for t, _, c in CHECKPOINT_TAGS}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INITIAL_TAG = "gaussian_h100"


# ---- per-tag cache loader -------------------------------------------------


class TagCache:
    def __init__(self, tag: str):
        rec = torch.load(CACHE_DIR / f"{tag}.pt", map_location="cpu", weights_only=False)
        self.tag = tag
        self.f_t: torch.Tensor = rec["f_t"]
        self.f_tp1: torch.Tensor = rec["f_tp1"]
        self.z_eval: torch.Tensor = rec["z_eval"]
        self.motion: torch.Tensor = rec["motion"]
        self.branch: str = rec["branch"]
        self.codes: Optional[torch.Tensor] = rec.get("codes")

        # 2D PCA of z_eval. Pad with a zero column if the latent collapsed.
        z = self.z_eval.numpy().astype(np.float64)
        z_centered = z - z.mean(axis=0, keepdims=True)
        n_components = min(2, z_centered.shape[1])
        try:
            self.pca = PCA(n_components=n_components).fit(z_centered)
            proj = self.pca.transform(z_centered)
        except Exception:
            proj = np.zeros((len(z_centered), n_components))
        if proj.shape[1] == 1:
            proj = np.concatenate([proj, np.zeros_like(proj)], axis=1)
        rng = np.random.default_rng(0)
        proj = proj + rng.normal(0, 0.005 * (np.std(proj) + 1e-9), size=proj.shape)
        self.proj_2d = proj

        # Order transitions by motion magnitude — the slider semantics.
        self.motion_order: np.ndarray = np.argsort(self.motion.numpy())

        self._model = None

    def df(self, current_idx: Optional[int] = None) -> pd.DataFrame:
        n = len(self.motion)
        df = pd.DataFrame({
            "pc1": self.proj_2d[:, 0].astype(float),
            "pc2": self.proj_2d[:, 1].astype(float),
            "motion": self.motion.numpy().astype(float),
            "transition_id": np.arange(n).astype(int),
        })
        if self.codes is not None:
            df["code_id"] = self.codes.numpy().astype(int)
        return df

    @property
    def model(self):
        if self._model is None:
            self._model = _load_lam(str(CKPT_DIR / TAG_TO_CKPT[self.tag]), DEVICE)
            self._model.eval()
        return self._model


_CACHES: dict[str, TagCache] = {}


def get_cache(tag: str) -> TagCache:
    if tag not in _CACHES:
        _CACHES[tag] = TagCache(tag)
    return _CACHES[tag]


def title_for(tag: str) -> str:
    c = get_cache(tag)
    extra = ""
    if c.branch == "vq" and c.codes is not None:
        extra = f"  ·  unique codes here: {int(c.codes.unique().numel())}/64"
    return f"{TAG_TO_LABEL[tag]}{extra}"


def frame_path(idx: int, which: str) -> str:
    return str(FRAMES_DIR / f"{idx:05d}_{which}.jpg")


def transition_info(c: TagCache, idx: int, percentile: float) -> str:
    parts = [
        f"transition **#{idx}**",
        f"motion **{c.motion[idx].item():.2f}**",
        f"motion percentile **{percentile:.0f}%**",
    ]
    if c.branch == "vq" and c.codes is not None:
        parts.append(f"code id **{int(c.codes[idx].item())}**")
    return "  ·  ".join(parts)


# ---- interpolation --------------------------------------------------------


@torch.no_grad()
def interpolate(tag: str, anchor_a, anchor_b, alpha: float):
    if anchor_a is None or anchor_b is None:
        return None, "pin two transitions as A and B (use the buttons), then drag"
    c = get_cache(tag)
    z_a = c.z_eval[anchor_a].to(DEVICE)
    z_b = c.z_eval[anchor_b].to(DEVICE)
    z = (1.0 - alpha) * z_a + alpha * z_b
    f_t = c.f_t[anchor_a].to(DEVICE)
    pred = c.model.fdm(torch.cat([f_t, z], dim=-1))

    f_tp1_all = c.f_tp1.to(DEVICE)
    dists = (f_tp1_all - pred).pow(2).sum(dim=-1)
    nearest = int(dists.argmin().item())
    near_dist = float(dists[nearest].sqrt().item())

    f_t_all = c.f_t.to(DEVICE)
    far_dist = float((f_t_all - pred).pow(2).sum(dim=-1).min().sqrt().item())

    info = (
        f"α = {alpha:.2f}  ·  retrieved transition #{nearest}  ·  "
        f"d(pred, nearest f_{{t+1}}) = {near_dist:.2f}  ·  "
        f"d(pred, nearest f_t) = {far_dist:.2f}"
    )
    return frame_path(nearest, "tp1"), info


# ---- UI -------------------------------------------------------------------


N = 500  # transitions per checkpoint (must match precompute)


with gr.Blocks(title="LAM Latent Action Tour", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# LAM latent action tour\n"
        "\n"
        "Drag the **motion percentile** slider to walk through 500 held-out "
        "transitions sorted from low motion (stationary worker) to high "
        "motion (active reaching, lifting). At each position, both the input "
        "frame `f_t` and the next frame `f_t+1` are real frames from a "
        "held-out video the LAM never saw.\n"
        "\n"
        "Switch the **branch + scale** dropdown to see the same data through "
        "any of the 6 trained LAMs. PCA is fit per-checkpoint; what matters "
        "is the *structure* (motion gradient, codebook clusters)."
    )

    state_anchor_a = gr.State(None)
    state_anchor_b = gr.State(None)

    with gr.Row():
        tag_dropdown = gr.Dropdown(
            choices=[label for _, label, _ in CHECKPOINT_TAGS],
            value=TAG_TO_LABEL[INITIAL_TAG],
            label="branch + data scale",
            interactive=True,
            scale=3,
        )

    # ---- TOUR (primary) ----

    initial_c = get_cache(INITIAL_TAG)
    initial_pct = 50
    initial_idx = int(initial_c.motion_order[(initial_pct / 100) * (N - 1) // 1] if False else
                      initial_c.motion_order[int((initial_pct / 100) * (N - 1))])

    pct = gr.Slider(
        0, 100, value=initial_pct, step=0.2,
        label="motion percentile  (drag to walk through transitions)",
    )
    info_md = gr.Markdown(transition_info(initial_c, initial_idx, initial_pct))

    with gr.Row():
        frame_t_img = gr.Image(
            value=frame_path(initial_idx, "t"),
            label="frame_t (input to IDM)",
            height=260,
        )
        frame_tp1_img = gr.Image(
            value=frame_path(initial_idx, "tp1"),
            label="frame_t+1 (target of FDM)",
            height=260,
        )

    state_idx = gr.State(initial_idx)

    with gr.Row():
        scatter = gr.ScatterPlot(
            value=initial_c.df(initial_idx),
            x="pc1",
            y="pc2",
            color="motion",
            title=title_for(INITIAL_TAG),
            height=380,
            tooltip=["transition_id", "motion"],
        )

    # ---- INTERPOLATION (secondary) ----

    with gr.Accordion("interpolate between two transitions (advanced)", open=False):
        gr.Markdown(
            "Pin the current transition as A or B, switch transitions, pin "
            "the other, then drag α. We linearly interpolate z and retrieve "
            "the nearest real `f_t+1` for the FDM's prediction."
        )
        with gr.Row():
            pin_a_btn = gr.Button("📌 pin current as A")
            pin_b_btn = gr.Button("📌 pin current as B")
            pick_pair_btn = gr.Button("🎲 pick contrasting pair", variant="primary")
            swap_btn = gr.Button("swap A ↔ B")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**A**")
                anchor_a_img = gr.Image(label="A: frame_t+1", height=180)
            with gr.Column():
                gr.Markdown("**B**")
                anchor_b_img = gr.Image(label="B: frame_t+1", height=180)
            with gr.Column():
                gr.Markdown("**Interpolated** — (1−α)·z_A + α·z_B → FDM → nearest real")
                interp_img = gr.Image(label="retrieved frame_t+1", height=180)
        with gr.Row():
            alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.02, label="α  (A → B)", scale=4)
            play_btn = gr.Button("▶ play A → B", variant="primary", scale=1)
        interp_info = gr.Markdown("pin both A and B, then drag the slider — or hit ▶ play")

    # ---- wiring ----

    def _idx_for_pct(c: TagCache, p: float) -> int:
        rank = int(round((p / 100.0) * (N - 1)))
        rank = max(0, min(N - 1, rank))
        return int(c.motion_order[rank])

    def on_pct_change(label, p):
        tag = LABEL_TO_TAG[label]
        c = get_cache(tag)
        idx = _idx_for_pct(c, p)
        return (
            frame_path(idx, "t"),
            frame_path(idx, "tp1"),
            transition_info(c, idx, p),
            idx,
        )

    pct.change(
        on_pct_change,
        inputs=[tag_dropdown, pct],
        outputs=[frame_t_img, frame_tp1_img, info_md, state_idx],
    )

    def on_tag_change(label, p):
        tag = LABEL_TO_TAG[label]
        c = get_cache(tag)
        idx = _idx_for_pct(c, p)
        return (
            frame_path(idx, "t"),
            frame_path(idx, "tp1"),
            transition_info(c, idx, p),
            idx,
            gr.ScatterPlot(value=c.df(idx), title=title_for(tag), x="pc1", y="pc2", color="motion"),
            None, None,                      # clear A and B
            None, None,                      # clear anchor imgs
            None, "pin both A and B, then drag the slider",
        )

    tag_dropdown.change(
        on_tag_change,
        inputs=[tag_dropdown, pct],
        outputs=[
            frame_t_img, frame_tp1_img, info_md, state_idx,
            scatter,
            state_anchor_a, state_anchor_b,
            anchor_a_img, anchor_b_img,
            interp_img, interp_info,
        ],
    )

    def _decode_evt_idx(evt: gr.SelectData, c: TagCache) -> Optional[int]:
        ev_idx = getattr(evt, "index", None)
        ev_val = getattr(evt, "value", None)
        coord = None
        if isinstance(ev_idx, (list, tuple)) and len(ev_idx) == 2 and all(
            isinstance(v, (int, float)) for v in ev_idx
        ):
            coord = (float(ev_idx[0]), float(ev_idx[1]))
        elif isinstance(ev_val, (list, tuple)) and len(ev_val) == 2 and all(
            isinstance(v, (int, float)) for v in ev_val
        ):
            coord = (float(ev_val[0]), float(ev_val[1]))
        if coord is not None:
            x, y = coord
            d = (c.proj_2d[:, 0] - x) ** 2 + (c.proj_2d[:, 1] - y) ** 2
            return int(np.argmin(d))
        if isinstance(ev_idx, int):
            return ev_idx
        if isinstance(ev_val, dict) and "transition_id" in ev_val:
            return int(ev_val["transition_id"])
        return None

    def on_scatter_select(label, evt: gr.SelectData):
        tag = LABEL_TO_TAG[label]
        c = get_cache(tag)
        idx = _decode_evt_idx(evt, c)
        if idx is None or not (0 <= idx < N):
            # leave everything as-is on a bad event
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        # find this idx's percentile rank
        rank = int(np.where(c.motion_order == idx)[0][0])
        new_pct = round(100.0 * rank / (N - 1), 1)
        return (
            frame_path(idx, "t"),
            frame_path(idx, "tp1"),
            transition_info(c, idx, new_pct),
            idx,
            new_pct,
        )

    scatter.select(
        on_scatter_select,
        inputs=[tag_dropdown],
        outputs=[frame_t_img, frame_tp1_img, info_md, state_idx, pct],
    )

    # ---- pin A/B + interpolation ----

    def _anchor_imgs(a, b):
        return (
            frame_path(a, "tp1") if a is not None else None,
            frame_path(b, "tp1") if b is not None else None,
        )

    def _pin(slot, idx, a, b):
        if slot == "A":
            a = idx
        else:
            b = idx
        a_img, b_img = _anchor_imgs(a, b)
        return a, b, a_img, b_img

    pin_a_btn.click(
        lambda idx, a, b: _pin("A", idx, a, b),
        inputs=[state_idx, state_anchor_a, state_anchor_b],
        outputs=[state_anchor_a, state_anchor_b, anchor_a_img, anchor_b_img],
    )
    pin_b_btn.click(
        lambda idx, a, b: _pin("B", idx, a, b),
        inputs=[state_idx, state_anchor_a, state_anchor_b],
        outputs=[state_anchor_a, state_anchor_b, anchor_a_img, anchor_b_img],
    )

    def _pick_pair(label):
        c = get_cache(LABEL_TO_TAG[label])
        order = c.motion_order
        a = int(order[int(0.05 * len(order))])
        b = int(order[int(0.95 * len(order))])
        return a, b, frame_path(a, "tp1"), frame_path(b, "tp1")

    pick_pair_btn.click(
        _pick_pair,
        inputs=[tag_dropdown],
        outputs=[state_anchor_a, state_anchor_b, anchor_a_img, anchor_b_img],
    )

    def _swap(a, b):
        a_img, b_img = _anchor_imgs(b, a)
        return b, a, a_img, b_img

    swap_btn.click(
        _swap,
        inputs=[state_anchor_a, state_anchor_b],
        outputs=[state_anchor_a, state_anchor_b, anchor_a_img, anchor_b_img],
    )

    def on_alpha(label, anchor_a, anchor_b, a):
        return interpolate(LABEL_TO_TAG[label], anchor_a, anchor_b, a)

    # Use .input() rather than .change() so the play generator's
    # programmatic alpha updates don't re-trigger this handler and race
    # the play stream.
    alpha.input(
        on_alpha,
        inputs=[tag_dropdown, state_anchor_a, state_anchor_b, alpha],
        outputs=[interp_img, interp_info],
    )

    def play_animation(label, anchor_a, anchor_b):
        """Stream the interpolation as an animation. Holds A's real
        frame_{t+1} for a beat, sweeps the FDM through alpha 0→1, then
        holds B's real frame_{t+1} for a beat."""
        import time
        if anchor_a is None or anchor_b is None:
            yield None, "▶ pin both A and B first", 0.0
            return
        tag = LABEL_TO_TAG[label]
        # 1) Real A frame
        yield frame_path(anchor_a, "tp1"), f"▶ start: A (transition #{anchor_a}, real frame)", 0.0
        time.sleep(0.6)
        # 2) FDM-driven sweep through alpha
        n_steps = 24
        for i in range(0, n_steps + 1):
            alpha_val = i / n_steps
            img, info = interpolate(tag, anchor_a, anchor_b, alpha_val)
            yield img, "▶ playing  ·  " + info, alpha_val
            time.sleep(0.10)
        # 3) Real B frame
        yield frame_path(anchor_b, "tp1"), f"▶ end: B (transition #{anchor_b}, real frame)", 1.0

    play_btn.click(
        play_animation,
        inputs=[tag_dropdown, state_anchor_a, state_anchor_b],
        outputs=[interp_img, interp_info, alpha],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
