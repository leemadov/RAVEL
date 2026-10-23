# RAVEL V2.1 — measured results

Everything here was measured in this repository on a cloud container with **4 CPU cores, 15 GB RAM and no GPU**
(PyTorch 2.9.0+cpu, Python 3.13). The target GPU (RTX 5070, 12 GB) was not available, so **no GPU, VRAM or
GPU-latency claim is made**. Raw outputs are in `results/` and training logs in `results/train_logs/`. Commands are in
`scripts/run_experiments.sh`, `scripts/jobs_main.txt` and `scripts/jobs_rest.txt`.

Because of the CPU budget, the ablations use **RAVEL-Micro**: the same architecture with smaller widths, about 8 M
parameters (`ravel/config.py:micro_overrides`). Only Stage 0, Stage 1 and the latency benchmark use RAVEL-Tiny (51.2 M).

## Summary

| question | answer measured here |
|---|---|
| Do all modules train with valid gradients? | **Yes.** 34 unit tests cover gradient flow to every module, causality, streaming ≡ batch, checkpoint round trip and resume. |
| Does RAVEL-Tiny fit the hardware budget? | **On CPU, yes:** training B=8 × 2 s peaks at about 2.3–3.3 GB of process RSS. **Not tested on a GPU.** |
| Does it produce intelligible speech? | **Not shown.** After 400 CPU steps the next-token CE is 5.55 nats (uniform guessing is 6.93), far from intelligible. Stage 2 below tests a narrow toy task. |
| Does it learn a nontrivial spoken input → response mapping? | See §5 (digit-successor toy task, audio only). |
| Does the RIF produce measurable listener predictions? | **Yes.** On held-out speakers: user-VAD AUC 0.877, barge-in AUC 0.74, turn-abandon AUC 0.79–0.81, overlap AUC 0.80 (§3). |
| Does the RIF improve conversational behaviour (the main question)? | **In the simulator, it beat both no-RIF baselines on all 3 seeds.** The gain is 36% lower interaction cost than a scalar cost critic: 7.12 ± 0.39 vs 11.11 ± 1.22. With 3 seeds this is not statistically significant (Welch p = 0.07; paired p = 0.11). It does **not** beat a hand-written heuristic (7.56), and the simulator's listener is rule-based, not human (§3). |
| Does the planner respond to incoming audio? | **Yes.** Revisions of uncommitted slots rise about 12× in the decision after a barge-in. **No behavioural benefit of the planner has been shown** (§4). |
| Real-time on CPU? | See §6. |
| Is RAVEL better than a conventional AR audio model? | **No** on next-token prediction: Baseline A (49.3 M) reached CE 5.44 vs 5.55 at equal steps, and runs 1.9× faster per step (§2). |

## 0. Stage 0 — architecture validation (RAVEL-Tiny, CPU, synthetic tensors)

`scripts/stage0_validate.py`, `results/stage0_cpu.json`. Parameters: APF 8.07 M, core 14.37 M, RIF 6.16 M, policy
0.86 M, planner 8.94 M, DNAG 11.07 M, feedback and action embedding 1.72 M; **total 51.21 M trainable**. The frozen
EnCodec (14.85 M) is excluded.

| batch × audio | fwd+bwd+AdamW step | process max RSS |
|---|---|---|
| 2 × 1 s | 1.72 s | 1.34 GB |
| 4 × 2 s | 4.57 s | 1.79 GB |
| 8 × 2 s | 5.50 s | 2.25 GB |

The checkpoint round trip gives bit-identical outputs (586 MB file with AdamW state). `scripts/check_gpu.py` reports
`cuda_available: false` in this container. The RTX 5070 / sm_120 compatibility check must be run on the target machine.

## 1. Data

* LibriSpeech dev-clean (CC BY 4.0): 5.12 h after trimming, 40 speakers, resampled to 24 kHz, EnCodec 4 × 1024 tokens.
  Split: AI voice = speaker 3576, 31 training users, 8 held-out test users.
* DuplexSim logs: 2,000 training episodes and 200 test episodes, 8 s each, built from real speech with a rule-based
  listener (`training/duplex_sim.py`). The logging policy is an observation-only heuristic with 10% exploration, mean
  cost 24.5 per episode. A second set of 2,000 episodes comes from a *confounded* logger that peeks at hidden
  turn-completion and user type.
* FSDD (CC BY-SA 4.0) for the digit toy task (§5).

## 2. Stage 1 — acoustic learning, RAVEL-Tiny vs. Baseline A (single seed, 400 steps, batch 8 × 2 s)

Both models predict the same tokens (the next 40 ms slot of the AI channel, 4 codebooks, EnCodec 75 Hz) from the same
inputs: own-channel history plus a microphone carrying echo, noise and, half of the time, another talker. The test set is
8 fixed batches from the 8 held-out speakers.

| model | trainable params | step | held-out CE, all 4 codebooks (nats/token) | CE codebook 0 | token accuracy |
|---|---|---|---|---|---|
| (uniform guess) | — | — | 6.93 | 6.93 | 0.1% |
| Baseline A (75 Hz LSTM ×4, d=1024, + depth GRU) | 49.35 M | 0 / 100 / 200 / 300 / **400** | 7.44 / 5.81 / 5.56 / 5.47 / **5.44** | 7.41 / 5.39 / 5.13 / 5.03 / **4.98** | **7.6%** |
| RAVEL-Tiny (full model) | 51.21 M | 0 / 100 / 200 / 300 / **400** | 7.46 / 5.98 / 5.70 / 5.59 / **5.55** | 7.48 / 5.67 / 5.40 / 5.28 / **5.22** | **6.8%** |

Training speed (1 thread, with other jobs sharing the 4 cores): Baseline A 4.9 s/step, RAVEL 9.3 s/step. Max RSS 2.35 vs
3.34 GB.

*Reading:* the conventional streaming AR LSTM is slightly better per step and about 2× faster per step. RAVEL also
learns echo-robust human VAD (train BCE 0.033) and APC, which the baseline does not. That is not evidence of better
acoustic modelling. Neither model is anywhere near intelligible speech after this much training.

## 3. Main question — does the Reciprocal Influence Field help? (DuplexSim, RAVEL-Micro, 3 seeds × 1,000 steps)

**Setup.** The three variants are identical except for the listener-modelling component:
* **B** — RAVEL without RIF. Actions come from a policy head trained by behaviour cloning on the logs. Plain argmax
  never speaks (see SPEC §5), so B's speak-threshold θ is **tuned on 60 validation episodes from training speakers**.
* **Bcrit** — the RIF architecture with **only** the scalar cost-to-go head, i.e. a model-free action-value critic.
* **C** — the full RIF: per-slot user-VAD, onset, overlap and listener-event heads plus the cost head, with ensemble
  uncertainty.

Bcrit and C act greedily on `cost + β·std`; β ∈ {0, 0.5, 1} is tuned on the same validation episodes. All models
run through the real streaming path (20 ms ticks) on **200 held-out-speaker episodes**. The AI's speech *content*
comes from the simulator bank, so this experiment tests timing, rate and backchannel decisions, not the generator.

**Interaction cost per 8 s episode (lower is better)**, mean ± SEM over 3 seeds (`results/e1_closed_loop.json`):

| policy | cost | overlap slots | gap slots (waiting for AI) | dead-air slots | barge-ins | turn abandons | re-prompts | response latency (slots) | backchannel share |
|---|---|---|---|---|---|---|---|---|---|
| logging policy (ε = 0.1) — training data | 24.71 | 17.6 | 11.0 | 4.9 | 0.59 | 0.51 | 0.01 | 2.4 | 0.11 |
| random | 72.34 | 57.3 | 16.4 | 2.5 | 0.34 | 0.70 | 0.03 | 0.7 | 0.20 |
| always silent | 17.95 | 0 | 77.0 | 11.5 | 0 | 0 | 1.94 | — | 0 |
| hand-written heuristic (ε = 0) | 7.56 | 3.2 | 5.3 | 6.9 | 0.57 | 0.47 | 0 | 3.3 | 0.03 |
| *oracle (sees hidden state; not a baseline)* | *0.59* | 0 | 0.0 | 6.8 | 0.13 | 0 | 0 | 0.0 | 0.02 |
| B, greedy | 17.95 ± 0.00 | 0 | 77.0 | 11.5 | 0 | 0 | 1.94 | — | 0 |
| B, sampled | 23.57 ± 0.08 | 16.8 | 11.0 | 5.1 | 0.55 | 0.49 | 0.00 | 2.8 | 0.11 |
| **B, tuned threshold** | **14.44 ± 1.23** | 4.3 | 39.1 | 13.7 | 0.29 | 0.15 | 0.77 | 9.6 | 0 |
| **Bcrit (scalar critic), tuned β** | **11.11 ± 1.22** | 0.6 | 38.6 | 6.3 | 0.35 | 0.21 | 0.92 | 1.6 | 0.05 |
| **C (full RIF), tuned β** | **7.12 ± 0.39** | 0.8 | 19.7 | 2.8 | 0.44 | 0.29 | 0.27 | 0.7 | 0.05 |

Per seed (tuned): C = 6.70 / 6.77 / 7.89; Bcrit = 9.83 / 13.54 / 9.95; B = 13.56 / 16.86 / 12.89.
C < Bcrit on 3 of 3 seeds: Welch t = 3.12, p = 0.07; paired t = 2.80, p = 0.11. C < B: Welch p = 0.02. Bcrit < B:
paired p = 0.005. Results for β = 0.5 without tuning are nearly identical (C 7.08 ± 0.43, Bcrit 11.15 ± 1.13).

**What this does and does not show.**
* Learning action-conditioned outcome models from logged data and acting greedily on them (Bcrit, C) beats imitating
  the logger (B) and the logger itself by a wide margin. This is the expected one-step policy-improvement effect.
* Adding explicit **listener-behaviour prediction** (C vs. Bcrit) lowered cost further on every seed. The effect is
  mostly fewer response gaps and re-prompts (19.7 vs. 38.6 gap slots) and less dead air (backchannels). C pays a small
  price in overlap, barge-ins and turn abandons. With 3 seeds the difference is suggestive, not significant.
* C does **not** beat a simple hand-written heuristic (7.12 vs. 7.56, p = 0.37), and it is far from the privileged
  oracle (0.59).
* **No listener-type adaptation was learned.** The simulator rewards matching speaking rate to the user's (hidden)
  type, but no model changes its rate with the user type. Seed 2 of C always speaks fast and gets about 10× more
  barge-ins from slow users than from fast ones (`results/rate_analysis.json`).
* The listener is a hand-written rule system built on real speech audio. **None of this establishes anything about
  human listeners.**

**Predictive quality on held-out logs** (64 held-out test episodes, taken actions only, step 1000, seeds 0/1/2):

| | C (full RIF) | Bcrit |
|---|---|---|
| cost-to-go MAE | 1.539 / 1.596 / 1.557 | 1.571 / 1.618 / 1.584 |
| cost-to-go Pearson r | 0.384 / 0.321 / 0.373 | 0.351 / 0.300 / 0.328 |
| user-VAD AUC (12 future slots) | 0.877 / 0.876 / 0.878 | — |
| user-VAD Brier / ECE | 0.142 / 0.023–0.040 | — |
| barge-in / turn-abandon / overlap AUC | 0.74–0.75 / 0.79–0.81 / 0.80 | — |

C predicts the scalar cost-to-go slightly better than Bcrit on every seed, so the auxiliary listener targets help the
value estimate a little. The cost target is a single Monte-Carlo return and very noisy, which keeps r low for both.
Re-prompt AUC is undefined because the test logs contain almost no re-prompts. The APF's echo-robust human VAD reaches
a tick-level AUC of 0.996 on held-out speakers with the AI's own echo present (all variants).

### E2 — are the action-conditioned predictions right *under intervention*?
__E2_SECTION__

### E3 — confounded logs
__E3_SECTION__

## 4. Planner responsiveness (criterion 31)

`evaluation/planner_analysis.py`, C seed 0, 100 held-out episodes, 900 user onsets (`results/planner_analysis_C_s0.json`).
At the first decision after a user onset, the mean squared revision of uncommitted plan slots k ≥ 1 rises from about
0.005 to 0.017. When the onset happens *while the AI is speaking* (barge-in, n = 96), it rises from about 0.008 to
0.097, and the plan's predicted AI activity for the next 280 ms drops from 0.85 to 0.19 within that one decision (40 ms).
**Caveat:** in this setup the selected *action* gates audio output and also conditions the planner. The plan is
therefore responsive, but we have not shown that it adds any behavioural benefit over a direct slot head.

## 5. Stage 2 toy task — spoken digit successor (audio only)
__DIGIT_SECTION__

## 6. Latency (CPU, measured; GPU not available)
__LATENCY_SECTION__

## 7. Not done / not demonstrated

* No GPU run of any kind. RTX 5070 compatibility, VRAM use and GPU latency are **untested**.
* No training on real two-channel conversational recordings, and no spoken QA beyond the digit toy task. Stages 2, 3 and
  5 on real dialogue are implemented (`training/dialog_data.py`, `training/train_conversation.py`) but **not run**.
* No intelligibility (ASR/WER) or prosody-quality evaluation. Stage 6 joint fine-tuning is implemented
  (`training/train_joint.py`) but **not run**.
* No long-conversation consistency test. The episodes are 8 s.
* Seeds: 3 per variant in the main ablation, and 1 for Stage 1 and the digit task.
