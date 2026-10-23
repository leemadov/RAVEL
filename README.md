# RAVEL V2.1 — Reciprocal Acoustic Variational Exchange Lattice

An experimental, speech-native, full-duplex conversational architecture in PyTorch. The research question is
whether an **action-conditioned model of the listener**, the *Reciprocal Influence Field* (RIF), improves
turn-taking behaviour beyond recurrence plus acoustic planning.

> **Status (read this first).** RAVEL-Tiny is implemented and tested, and its components have been trained in
> short CPU experiments. It is **not** a conversational AI: it has not been shown to produce intelligible
> open-domain speech or to answer real spoken questions. All results below were measured in this repository on a
> 4-core CPU container with **no GPU**. The target RTX 5070 was not available, so nothing here has been run on it.
> See [RESULTS.md](RESULTS.md) for every number and its limitations.

## What is in the box

| subsystem | file | what it does |
|---|---|---|
| A. Acoustic Perception Field | `ravel/acoustic_encoder.py` | causal log-mel (20 ms ticks) → dilated GLU causal convs + GRU. Takes the loudspeaker reference as an input; echo-robust human-VAD head; APC future-mel head |
| B. Dual-State Conversational Core | `ravel/conversational_core.py` | coupled RG-LRU-style gated linear recurrences for **H**uman, **I**nteraction and **A**I states, plus a clocked slow memory (200 ms). No Transformer |
| C. Reciprocal Influence Field | `ravel/influence_field.py` | f(H, A, I, M, candidate action) → per-slot user-VAD, onset timing, overlap, listener events and cost-to-go, with a 4-member bootstrapped ensemble for uncertainty |
| policy | `ravel/policy.py` | score = E[cost] + w·P(overlap) + β·std + κ·compute − α·relevance; argmax, or straight-through Gumbel in training |
| D. Short-Horizon Acoustic Planner | `ravel/acoustic_planner.py` | 8 editable 40 ms slots; shift → revise (gated) → commit slot 0; conditioned on the RIF's prediction |
| E. Direct Neural Audio Generator | `ravel/audio_generator.py` | autoregressive GRU over the 3 frames × 4 codebooks of each slot → frozen EnCodec 24 kHz decoder |
| F. Self-Monitoring Feedback | `ravel/feedback.py`, `inference/echo.py` | emitted tokens and action fed back into the core; PBFDAF echo canceller |
| Baseline A | `ravel/baselines.py` | parameter-matched conventional streaming AR codec LM (75 Hz LSTM + depth GRU) |
| streaming | `inference/streaming.py` | tick-by-tick full-duplex inference (listen every 20 ms, commit 40 ms) |
| simulator | `training/duplex_sim.py` | controlled two-party turn-taking world built from real LibriSpeech speech, with a rule-based reactive listener (ground truth known, interventions possible) |

RAVEL-Tiny has **51.2 M trainable parameters**. The frozen EnCodec (14.9 M) is excluded. The exact math and tensor
shapes are in [docs/SPEC.md](docs/SPEC.md). Prior work and novelty are covered in
[docs/PRIOR_ART.md](docs/PRIOR_ART.md); in short, it is a novel *combination* of known parts, and the closest prior
work is VAP used as a user model (Ekstedt et al. 2023). Data and licences are in
[docs/DATA_AND_LICENSES.md](docs/DATA_AND_LICENSES.md).

## Quick start

```bash
pip install -r requirements.txt          # encodec needs: pip install --use-pep517 encodec
python -m pytest -q tests                # 34 unit tests (CPU, ~1 min; codec tests download EnCodec weights)
python scripts/check_gpu.py              # verifies the installed torch build actually runs on your GPU (e.g. sm_120)
python scripts/stage0_validate.py --device cuda --amp bf16   # forward/backward, peak memory, checkpoint round-trip
bash scripts/run_experiments.sh          # every experiment in RESULTS.md (CPU: many hours)
```

On an RTX 50-series (Blackwell, sm_120) GPU, install a CUDA 12.8+ PyTorch build
(`pip install torch --index-url https://download.pytorch.org/whl/cu128`) and run `scripts/check_gpu.py` first. **This was
not tested here.**

## Repository layout

```
configs/ravel_tiny.yaml       RAVEL-Tiny configuration (micro config for ablations: ravel/config.py:micro_overrides)
ravel/                        model (six subsystems + baseline + codec wrapper)
training/                     losses, datasets, simulator, stage scripts (acoustic, influence, conversation, joint, digit QA)
inference/                    streaming engine, echo canceller, microphone / wav demo
evaluation/                   closed-loop ablations, counterfactual (interventional) evaluation, latency, planner analysis
scripts/                      data prep, Stage 0 validation, GPU check, experiment runner
tests/                        unit tests (shapes, causality, streaming equivalence, gradients, checkpointing, sim)
results/                      measured results (JSON) used in RESULTS.md
docs/                         SPEC, PRIOR_ART, DATA_AND_LICENSES
```
