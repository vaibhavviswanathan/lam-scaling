# LAM Crossover POC for Egocentric-100K

A minimal, runnable Latent Action Model with **two interchangeable bottlenecks** —
discrete (VQ-VAE) and continuous (Gaussian) — trained on Build AI's Egocentric-100K
dataset. Designed to fit on a 16GB GPU (e.g. Tensorbook 4090 mobile) and produce
a credible preliminary **crossover result** for a compute-grant application.

## What this is

A controlled head-to-head comparison: same encoder, same IDM, same FDM, same
optimizer, same compute per cell — only the bottleneck changes. The discrete
branch follows LAPO / Genie / GO-1; the continuous branch follows
DreamDojo / AdaWorld / GR00T N1. No paper has run this comparison at scale on
physical-world video.

```
       (frozen DINOv2-small)
frame_t    ─►  f_t  ──┐
                      ├─► IDM ──► z_pre ──► [BOTTLENECK] ──► z_used ──► FDM ──► f̂_{t+1}
frame_{t+1}─► f_{t+1}─┘                                                          │
                                                                                 ▼
                                                       loss = MSE(f̂_{t+1}, f_{t+1})
                                                            + bottleneck_loss

[BOTTLENECK] is one of:
  Discrete (VQ):       z_pre ──► nearest codebook entry z_q (K=64 codes)
                       loss   = β·||sg(z_q) − z_pre||² + ||z_q − sg(z_pre)||²
  Continuous (Gauss):  z_pre ──► (μ, log σ²) ──► z = μ + σ·ε  with ε ~ N(0, I)
                       loss   = β·KL(N(μ,σ²) || N(0,I))   with free-bits floor
```

The two branches expose `z_eval` (deterministic — `z_q` for VQ, `μ` for Gaussian)
so all downstream probes are apples-to-apples comparable.

- `data.py`     — Streaming WebDataset loader for Egocentric-100K
- `model.py`    — DINOv2 encoder + IDM + swappable bottleneck (VQ or Gaussian) + FDM
- `train.py`    — Single-cell training loop, `--bottleneck {vq,gaussian}`
- `eval.py`     — Held-out recon, motion probe, branch-specific diagnostics, optional dead-zone
- `hand_probe.py` — MediaPipe Hands linear probe (in-dist + OOD)
- `scaling.py`  — Crossover sweep: both branches × multiple data scales

## Quick start

```bash
pip install -r requirements.txt

# Discrete (VQ) branch
python train.py --bottleneck vq --hours 10 --steps 5000 --bs 16 --save lam_vq_h10.pt

# Continuous (Gaussian) branch — same flags + KL warmup
python train.py --bottleneck gaussian --hours 10 --steps 5000 --bs 16 --save lam_gauss_h10.pt

# Evaluate either checkpoint (auto-detects branch from config)
python eval.py --ckpt lam_vq_h10.pt --hours 1.0
python eval.py --ckpt lam_gauss_h10.pt --hours 1.0 --dead_zone

# Hand-pose probe — works on either branch
python hand_probe.py --ckpt lam_vq_h10.pt --source hf --num_clips 200
python hand_probe.py --ckpt lam_gauss_h10.pt --source dir --path /data/ego4d_sample

# Full crossover sweep (the grant-proposal headline figure)
python scaling.py --branches vq gaussian --hours 1 10 100 --steps 1500 4000 10000
```

## Compute budget for the POC

| Setup                | Cost     | Time  |
|---------------------|----------|-------|
| Tensorbook 4090M    | sunk     | overnight |
| Vast.ai A100 spot   | ~$10-30  | overnight |
| Lambda On-Demand H100 | ~$30-60 | overnight |
| Modal serverless A10G | ~$5-15  | weekend |

## What goes in the grant proposal

The headline artifact is the output of `scaling.py`:

1. `runs/scaling.png` — log-log plot of held-out reconstruction loss vs. data
   hours, **with both branches overlaid**. This is the crossover figure.
2. `runs/scaling.json` — the underlying numbers, including branch-specific
   diagnostics (codebook utilization for VQ, KL / active-dims / dead-zone for
   Gaussian).
3. `hand_probe.py` outputs at the largest checkpoints, on Egocentric-100K
   held-out (in-distribution) and a local Ego4D / EPIC-KITCHENS sample (OOD).
4. A one-paragraph caption: *"On Egocentric-100K subsets (1–100h), the
   discrete branch gives held-out MSE ≈ ___ at 100h with α ≈ ___, while the
   continuous branch gives ___ with α ≈ ___. OOD hand-pose R² on Ego4D at
   100h is ___ (discrete) vs. ___ (continuous). We propose to extend both
   curves through 1M hours on Egocentric-1M to identify the crossover regime."*

That converts "I'd like to do this" into "I've validated the method on both
branches; fund the extension." Reviewers fund momentum.

## Honest caveats

- **DINOv2 is an image encoder.** It was trained on internet images, not
  factory video. A fairer apples-to-apples scaling study would also include
  video pretraining (V-JEPA-2, VideoMAEv2). Note as future work.
- **Branch-specific failure modes to watch:**
  - *Discrete (VQ):* codebook collapse. Watch `codes_used / num_codes` —
    if it drops below ~30%, raise `--latent_dim`, lower `--num_codes`, or
    swap to EMA codebook updates.
  - *Continuous (Gaussian):* posterior collapse. Watch `active_dims` from
    the training log. With `latent_dim=64` and `free_bits=0.5`, you should
    see active dims rise from 0 to roughly 20–50 during KL warmup. If it
    stays at 0, lower `--target_beta` (try 0.05). If it stays at 64, the
    bottleneck isn't doing anything — raise `--target_beta` (try 0.3).
- **Free bits is doing real work.** The Gaussian branch *will* posterior-
  collapse without free bits at small data scale. The default
  `free_bits=0.5` is conservative; tune up if collapse persists, down if
  the bottleneck has too little capacity.
- **The motion probe is a sanity check, not a real action probe.** Use
  `hand_probe.py` for the headline metric. Motion probe is included to
  catch obvious training failures cheaply.
- **"Hours seen" is approximate** (videos × ~3min). For a publication-
  quality curve, swap in actual frame counts via the dataset's
  `duration_sec` metadata field.

## Extensions for the actual paper (Phase 2 territory)

1. **Replace DINOv2 with V-JEPA-2** — same data, video-native pretraining,
   eliminates the image-encoder caveat above.
2. **Hand-pose probe at every cell**, not just the largest — full
   crossover-by-OOD-dataset plot.
3. **Parameter scaling sweep** — fix data, vary `--latent_dim` and MLP
   widths through `{50M, 200M, 500M, 1B}` trainable params per branch.
4. **World model on the winning branch** — train a video diffusion
   (continuous winner) or autoregressive transformer (discrete winner)
   conditioned on `(f_t, z_t) → f_{t+1}`. Generates an interactive
   simulator of factory work.
5. **Cross-embodiment transfer** — small action decoders on Open-X
   subsets (Franka, UR5, xArm) mapping `z_eval` to robot end-effector
   commands. Train on two embodiments, test on a held-out third.
6. **Codebook size sweep within the discrete branch** — does the optimal
   `K` depend on data scale? (LAPA's open question.)
