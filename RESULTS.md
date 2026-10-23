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
| Do all modules train with valid gradients? | **Yes.** 31 unit tests cover gradient flow to every module, causality, streaming ≡ batch, checkpoint round trip and resume. |
| Does RAVEL-Tiny fit the hardware budget? | **On CPU, yes:** training B=8 × 2 s peaks at about 2.3–3.3 GB of process RSS. **Not tested on a GPU.** |
| Does it produce intelligible speech? | **Only in a 10-word closed world.** Its generated digit answers are recognised by an independent classifier 86% of the time (§5). Open-vocabulary speech is **not** intelligible: Stage 1 next-token CE is 5.55 nats after 400 CPU steps (uniform guessing is 6.93), and no ASR/WER was measured. |
| Does it learn a nontrivial spoken input → response mapping? | **Yes, in a toy closed world.** It says the next digit aloud in 86% of 250 held-out spoken questions (chance 10%). Scoring is closed-loop and audio-only, by an independent classifier; the policy chooses its own timing (§5). |
| Does the RIF produce measurable listener predictions? | **Yes.** On held-out speakers: user-VAD AUC 0.877, barge-in AUC 0.74, turn-abandon AUC 0.79–0.81, overlap AUC 0.80 (§3). |
| Does the RIF improve conversational behaviour (the main question)? | **In the simulator, it beat both no-RIF baselines on all 3 seeds.** The gain is 36% lower interaction cost than a scalar cost critic: 7.12 ± 0.39 vs 11.11 ± 1.22. With 3 seeds this is not statistically significant (Welch p = 0.07; paired p = 0.11). It does **not** beat a hand-written heuristic (7.56), and the simulator's listener is rule-based, not human (§3). |
| Does the planner respond to incoming audio? | **Yes.** Revisions of uncommitted slots rise about 12× in the decision after a barge-in. **No behavioural benefit of the planner has been shown** (§4). |
| Real-time on CPU? | **No.** RAVEL-Tiny needs 46 ms of compute per 40 ms slot without the codec (1.15× real time) and 98 ms with streaming EnCodec decoding (2.44×) on 4 CPU cores. GPU not measured (§6). |
| Is the RIF's prediction causal? | **Partly testable here (§3 E2/E3).** With randomised logs, values for untaken actions are nearly as accurate as for taken ones. Trained on logs from a logger that sees hidden state, the same model gets worse in deployment on 3/3 seeds (8.49 vs 7.12, p = 0.049). |
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
`evaluation/counterfactual.py`, `results/e2_e3_counterfactual_summary.json`. On 40 held-out-speaker episodes we stop
every 5 slots (840 decisions per model), snapshot the simulator, and for **every** candidate action run 32 rollouts:
take the action, then follow the logging policy. This gives the true interventional `Q^{π_b}(h, a)`, which the cost
head estimates, and true listener-event and user-VAD probabilities. The Monte-Carlo SEM of the truth is about 0.31.

| state distribution | metric (mean of 3 seeds) | C (full RIF) | Bcrit |
|---|---|---|---|
| logging policy's states | value MAE: logged / untaken actions | 0.72 / 0.80 | 0.73 / 0.81 |
| | Spearman across the 5 actions | 0.53 | 0.55 |
| | regret of greedy choice vs. true best | 0.152 | 0.143 |
| | *regret of the logger's own action* | *0.388* | *0.388* |
| | bias for speaking into a mid-turn pause | −1.08 | −0.84 |
| **each model's own (deployed) states** | value MAE, untaken actions | **0.86** | 0.91 |
| | Spearman across actions | **0.49** | 0.46 |
| | regret of its own choices | **0.158** (0.134 / 0.148 / 0.191) | 0.197 (0.193 / 0.188 / 0.210) |

Listener predictions under intervention (C): barge-in probability MAE 0.03, turn-abandon MAE 0.06. User-VAD
probability correlation with the truth falls from r = 0.88 for the next slot (40 ms ahead) to r = 0.51 twelve slots
(480 ms) ahead, with mean MAE 0.26.

*Reading.* Value estimates for actions the logger did **not** take are only modestly worse than for logged ones,
because the logger randomises timing and rate so the estimates are identifiable. Both learned critics choose actions
with less than half the regret of the logger's own choices. On the logger's states, the listener heads give **no**
counterfactual-accuracy advantage. On the states each policy actually visits, C's estimates and choices are somewhat
better on all 3 seeds. That is consistent with the auxiliary listener targets helping generalisation under the
distribution shift created by acting greedily, and it is the most plausible explanation of C's closed-loop advantage.
All models are over-optimistic about speaking into mid-turn hesitation pauses (true cost about 3.95, predicted about
2.8). That is partial observability: from audio alone a mid-turn pause can look like a turn end.

### E3 — confounded logs
Same architecture and recipe as C, but trained on 2,000 episodes from a **confounded** logger. That logger sees hidden
simulator state: it responds within 1–3 slots when the turn is truly complete, waits 12–25 slots otherwise, and matches
its speaking rate to the hidden user type 80% of the time. Deployed without hidden information, on the same 200 test
episodes (`results/e3_closed_loop.json`):

| model | cost | barge-ins | re-prompts | turn abandons | share of speech spoken slowly |
|---|---|---|---|---|---|
| C trained on randomised logs (E1) | **7.12 ± 0.39** (6.70 / 6.77 / 7.89) | 0.44 | 0.27 | 0.29 | ≈ 0% |
| C trained on confounded logs | 8.49 ± 0.26 (8.33 / 9.00 / 8.15) | 0.79 | 0.44 | 0.28 | ≈ 91% |
| *confounded logger itself (privileged, ε = 0.1)* | *22.07* | 0.39 | 0.01 | 0.42 | — |

The confounded-log model is worse on 3 of 3 seeds (Welch p = 0.049). Its main failure is that it speaks **slowly to
everyone**: in the confounded logs slow speech was mostly used with slow users, for whom it is safe. Fast users then
barge in 1.37 times per episode, versus 0.13 for slow users (`results/rate_analysis_e3.json`). One-step interventional
errors of its value head, measured against its own target `Q^{π_conf}`, are *not* larger than in E1 (regret 0.15, rate
biases within ±0.5). Our reading is that the damage comes from the target itself. `Q^{π_conf}` assumes the informed
logger will handle the following slots, for example by switching to the matched rate. Rate choices therefore look
nearly equivalent one step ahead, but they compound when a policy without that information repeats them every slot.
The confounded logs also cover the action space differently, so this experiment cannot fully separate confounding from
coverage. **Either way, it is direct evidence that an action-conditioned predictor learned from observational
conversations is not a reliable guide to intervention when the logged speaker had information the model lacks.**

## 4. Planner responsiveness (criterion 31)

`evaluation/planner_analysis.py`, C seed 0, 100 held-out episodes, 900 user onsets (`results/planner_analysis_C_s0.json`).
At the first decision after a user onset, the mean squared revision of uncommitted plan slots k ≥ 1 rises from about
0.005 to 0.017. When the onset happens *while the AI is speaking* (barge-in, n = 96), it rises from about 0.008 to
0.097, and the plan's predicted AI activity for the next 280 ms drops from 0.85 to 0.19 within that one decision (40 ms).
**Caveat:** in this setup the selected *action* gates audio output and also conditions the planner. The plan is
therefore responsive, but we have not shown that it adds any behavioural benefit over a direct slot head.

## 5. Stage 2 toy task — spoken digit successor (audio only)
`training/digit_qa.py`. Task: a user says a digit d (FSDD; 5 user speakers, held-out *recordings*, takes 40–49).
RAVEL must answer **(d + 1) mod 10 aloud** in a fixed AI voice (FSDD speaker "jackson") after the user finishes.
The model is RAVEL-Micro (RIF disabled, action set {silent, speak}). It was trained for 1,500 steps (batch 32 × 1.92 s,
single seed) on mic log-mels → EnCodec tokens, with no text anywhere.

**Evaluation is closed-loop and audio-only.** The model listens tick by tick. Its policy head decides when to start and
stop speaking. The DNAG samples tokens (temperature 0.7, top-k 50) and the frozen codec decodes them in streaming
fashion; the microphone also hears the echo of the model's own generated audio. The first 1 s of generated speech is
then classified by an **independent** CNN digit classifier, trained only on real recordings. That classifier scores
99.3% on held-out real recordings, 96.7% after codec round-trip, and 100% on held-out AI-voice recordings after codec
round-trip (n = 300).

| mode (250 held-out questions) | answered | **correct successor digit** | 95% CI | repeated the user's digit | start delay after user stops |
|---|---|---|---|---|---|
| policy decides timing | 100% | **86.0%** (215/250) | 81.1–90.1% | 3.2% | 5.4 slots (≈ 220 ms; trained gap 4–6 slots) |
| timing forced to the reference start | 100% | 86.0% | 81.1–90.1% | 0.4% | 5.0 slots |
| chance | — | 10% | — | 10% | — |

Per-target accuracy ranges from 68% (answer "4") to 96%. Predicted labels are spread over all ten digits (20–37 per
class), so the model is not collapsing to one answer. A teacher-forced probe, which ranks the 10 candidate answers by the
DNAG's cross-entropy, picks the correct successor 37% of the time (45.5% on the first 3 answer slots; chance 10%;
binomial p < 1e-23). That probe underestimates ability, because each candidate is a *random* take whose own
predictability dominates the score. Stereo examples (left = user, right = RAVEL) are in
`results/digit_qa_policy_examples/`.

**What this shows.** A nontrivial spoken input → spoken response mapping (success criterion 28) was learned
end-to-end through the audio-native path, with learned turn timing, in a tiny closed world: 10 one-word questions, a
fixed answer voice, and user speakers seen in training (only the recordings are new). **It is not open-domain spoken
QA**, and the answers mostly reproduce the AI speaker's training recordings. The 86% depends on that classifier and on
1 s windows.

## 6. Latency (CPU, measured; GPU not available)
`evaluation/latency.py`. Protocol: batch 1, fp32, `torch.inference_mode`, 4 threads on the 4-core container, 50
warm-up ticks discarded, speaking forced on every slot so the generator and codec always run (worst case).
Untrained weights; compute time does not depend on training.

| configuration | listen tick (every 20 ms) | decision tick (every 40 ms) | compute per 40 ms slot | real-time factor |
|---|---|---|---|---|
| RAVEL-Tiny, no codec decode | 11.6 ms | 34.4 ms | 46.0 ms | **1.15** (not real time) |
| RAVEL-Tiny + EnCodec streaming decode (75-frame context) | 13.7 ms | 84.0 ms (p95 116) | 97.8 ms | **2.44** |
| RAVEL-Micro + EnCodec decode | 4.8 ms | 59.9 ms | 64.7 ms | 1.62 |

EnCodec re-decoding with context costs 37 ms per slot at 75 frames, 17.5 ms at 15 frames and 10.3 ms at 3 frames; the
SNR vs. full decoding is 27.6 / 24.9 / 17.9 dB. A truly stateful streaming decoder (not implemented) would remove
most of this cost.

**Targets.** First-audio < 200 ms and interruption reaction < 100 ms were **not met on this CPU**, because compute
alone exceeds real time. The algorithmic floor is one slot boundary (40–60 ms) plus compute. In simulator *behaviour*,
the trained C policy starts speaking 0.7 slots (≈ 30 ms) after the user's turn becomes complete, and its plan
switches to silence within one decision of a barge-in; both are in slot time, not wall-clock time. GPU latency is
unmeasured. On a GPU, per-tick latency would likely be dominated by kernel-launch overhead from many small sequential
ops, which is a guess; CUDA graphs or a fused recurrent kernel would be needed.

## 7. Not done / not demonstrated

* No GPU run of any kind. RTX 5070 compatibility, VRAM use and GPU latency are **untested**.
* No training on real two-channel conversational recordings, and no spoken QA beyond the digit toy task. Stages 2, 3 and
  5 on real dialogue are implemented (`training/dialog_data.py`, `training/train_conversation.py`) but **not run**.
* No intelligibility (ASR/WER) or prosody-quality evaluation. Stage 6 joint fine-tuning is implemented
  (`training/train_joint.py`) but **not run**.
* No long-conversation consistency test. The episodes are 8 s.
* Seeds: 3 per variant in the main ablation, and 1 for Stage 1 and the digit task.
