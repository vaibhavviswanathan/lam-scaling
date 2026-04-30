# LAM latent action tour — web demo

Vanilla-HTML / vanilla-JS reimplementation of the Gradio demo. Same data
flow, more design control, faster client-side interactions, no Gradio
component quirks.

- **Backend** (`server.py`): FastAPI. Loads the precomputed cache, serves
  static assets, exposes a small JSON API. The FDM forward for live
  interpolation is ~1 ms on GPU.
- **Frontend** (`static/index.html`, `style.css`, `app.js`): vanilla
  modern CSS (Grid + Flexbox, system font stack, teal accent), vanilla JS
  for state, Plotly.js (CDN) for the scatter so clicks fire reliably.

## Run

```bash
# 1) Precompute the per-checkpoint cache (same script as the Gradio demo).
python scripts/precompute_demo_cache.py \
    --ckpt_dir runs/sweep \
    --data_path /home/vai/data/ego100k_heldout \
    --n_transitions 500

# 2) Launch the FastAPI server on http://127.0.0.1:8000
python webdemo/server.py
```

The first time the server hits a new checkpoint it computes both t-SNE
and UMAP projections of `z_eval` (~15 s per checkpoint on this hardware)
and persists them to
`runs/demo_cache/<tag>_projections.npz`. Subsequent server boots load
those projections instantly.

## What's in the UI

- **Branch + scale dropdown** — swaps in any of the 6 trained LAMs.
- **Tour card**: a motion-percentile slider and the corresponding
  `frame_t` / `frame_{t+1}` from held-out video. Underneath, a PCA /
  t-SNE / UMAP scatter of `z_eval` colored by motion. Click any point on
  the scatter to jump the tour slider to that transition.
- **Interpolation card**: pin the current transition as A or B (or hit
  🎲 contrasting pair for an instant low-vs-high-motion pair), then drag
  α — the "interpolated" pane shows the held-out frame whose `f_{t+1}`
  is closest to the FDM's prediction. The ▶ play button animates A → B
  with A's real frame at the start, ~25 retrieved frames in between, and
  B's real frame at the end.

## API surface

```
GET  /api/checkpoints                      → [{tag, label, ckpt}, ...]
GET  /api/cache/{tag}                      → all 500 transitions w/ pca/tsne/umap
POST /api/interpolate                      → live FDM forward + retrieval
```

`POST /api/interpolate` body:
```json
{"tag": "gaussian_h100", "anchor_a": 50, "anchor_b": 300, "alpha": 0.5}
```

Response:
```json
{"nearest_idx": 49, "alpha": 0.5,
 "d_pred_to_nearest_tp1": 16.0, "d_pred_to_nearest_t": 16.0}
```

## Why retrieval, not generation?

The LAM operates in DINOv2 feature space, not pixel space. There's no
RGB decoder here, so the "interpolated" frame is the held-out clip whose
`f_{t+1}` is closest to the FDM's prediction. This is honest about what
the model knows and avoids reading meaning into hallucinations a
generative decoder would invent. A pixel-space decoder is a Phase-2
deliverable per `proposal.md`.

## In-distribution vs OOD

These are 500 Egocentric-100K transitions from videos at HF stream
indices 2000–2049, while the LAMs were trained on indices 0–1999.
**The model has not seen these specific clips** — but they're from the
same dataset, same factories, same workers, same cameras. So this is a
legitimate "the LAM didn't memorize" test, but **not** the
out-of-domain validation the proposal's headline metric (Hand R² on
Ego4D / EPIC-KITCHENS) calls for.
