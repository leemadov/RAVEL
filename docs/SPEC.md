# RAVEL V2.1 — Mathematical specification (as implemented)

This document describes the code in `ravel/`. Shapes use the RAVEL-Tiny defaults (`configs/ravel_tiny.yaml`).
`B` = batch, `T` = number of 20 ms ticks, `S = T/2` = number of 40 ms slots, `D = S-1` = number of decisions,
`C` = number of candidate actions (7 in Tiny; 5 in the simulator), `E` = ensemble size (4), `N` = RIF horizon (12 slots),
`K` = plan slots (8), `F` = codec frames per slot (3), `Q` = codebooks (4), `V` = codebook size (1024).

## 1. Time base and causality

| quantity | value | samples @ 24 kHz |
|---|---|---|
| tick (perception / core update) | 20 ms | 480 |
| slot (decision / commit) | 40 ms = 2 ticks | 960 |
| EnCodec frame | 13.33 ms | 320 |
| slot ↔ codec | exactly 3 frames per slot | — |
| plan horizon | 8 slots = 320 ms | 7680 |
| RIF horizon | 12 slots = 480 ms | 11520 |

* Tick `t` covers samples `[480t, 480(t+1))`. Its log-mel is computed from the `n_fft = 1024` samples **ending** at
  `480(t+1)` (left zero padding), so no future audio is used (`ravel/features.py`, tested in `tests/test_modules.py`).
* Slot `s` = ticks `{2s, 2s+1}` = codec frames `{3s, 3s+1, 3s+2}`.
* **Decision `d = s`** happens after tick `2s+1`. It selects action `a_{s+1}` and commits audio for slot `s+1`. A user event
  at the start of slot `s` can therefore change output starting at slot `s+1`: algorithmic reaction latency is 40–60 ms
  plus compute, plus the codec decoder (§7). Already-emitted slots never appear in the plan, so they cannot be revised.
* End-to-end causality is tested: perturbing microphone audio from slot 4 onward leaves decisions 0–3 bit-identical
  (`tests/test_model.py::test_end_to_end_causality`). Streaming tick-by-tick inference reproduces the batched
  teacher-forced computation (`test_streaming_matches_teacher_forced`).

## 2. A. Acoustic Perception Field (`ravel/acoustic_encoder.py`)

Inputs per tick: `mic_mel_t, ref_mel_t ∈ R^80` (ref = what the loudspeaker plays = AI's own output).

```
u_t  = W_in [ (mic_mel_t+6)/4 ; (ref_mel_t+6)/4 ]                      (B, 384)
c_t  = GLUConv_8 ∘ ... ∘ GLUConv_1 (u_{≤t})                           kernel 3, dilations 1,2,4,8,1,2,4,8
       GLUConv: x + GLU(Conv1d_{d→2d}(pad_left(RMSNorm(x))))
g_t  = GRU(c_t, g_{t-1})                                              fast acoustic memory
z_t  = RMSNorm(c_t + g_t)                                              (B, 384)
```
Receptive field of the conv stack: `1 + 2·(1+2+4+8)·2 = 61` ticks (1.22 s); the GRU extends it indefinitely.
Heads: `p_user_t = σ(w·z_t)` (human voice activity, echo-robust because `ref_mel` is an input) and
APC `ŷ_t = W z_t ≈ mic_mel_{t+3}`.

## 3. B. Dual-State Conversational Core (`ravel/conversational_core.py`)

Gated linear recurrent cell (RG-LRU style, De et al. 2024):
```
[x, r, i] = W u_t ;   log a_t = -c · σ(r) · softplus(-Λ)   (c = 8)
s_t = a_t ⊙ s_{t-1} + sqrt(1 - a_t²) ⊙ (σ(i) ⊙ x)
y_t = s_t + FFN(RMSNorm(s_t))
```
`Λ` is initialised so that `a^c` is uniform in a per-state range; time constant `τ = 1/(1-a)` ticks:

| state | dim | `a^c` init range | τ range |
|---|---|---|---|
| H (human) | 640 | [0.8, 0.995] | 100 ms – 4 s |
| I (interaction) | 320 | [0.6, 0.98] | 50 ms – 1 s |
| A (AI) | 640 | [0.9, 0.9995] | 200 ms – 40 s |
| M (slow memory) | 512 | GRU updated every 10 ticks | clocked, 200 ms |

Per tick (order matters; coupling projections `P_xy` map to 192-d):
```
H_t = Cell_H([z_t ; P_ah y^A_{t-1} ; P_ih y^I_{t-1} ; P_mh M_{t-1} ; p_user_t])
I_t = Cell_I([P_hi y^H_t ; P_ai y^A_{t-1} ; f_t ; e(a_s) ; p_user_t])
A_t = Cell_A([P_ha y^H_t ; P_ia y^I_t ; P_pa mean_k(P_k) ; P_ma M_{t-1} ; f_t (; intent_t)])
M_t = GRUCell(mean_{window}[y^H; y^A; y^I], M_{t-1})   if (t+1) mod 10 == 0 else M_{t-1}
```
`f_t` = self-monitoring feedback (§7), `e(a_s)` = embedding of the action governing the current slot, `P` = current plan.
`intent_t` (2-d, optional) is only used in the simulator, where the AI's response *content* is supplied externally; it tells
the AI whether it still has something to say (its own private knowledge). Because H, I, A are coupled through
previous-tick outputs, the core is a non-linear RNN and is stepped sequentially: `O(T)` time, `O(1)` state memory at inference.
No Transformer is used in the core.

## 4. C. Reciprocal Influence Field (`ravel/influence_field.py`)

```
c_t      = MLP([y^H_t ; y^A_t ; y^I_t ; M_t])                         (B, d=640)
γ, β     = W_film e(a)          (zero-initialised)                    (C, d)
h_t(a)   = ResMLP²( c_t ⊙ (1+γ(a)) + β(a) )                            (B, C, d)
o^e_t(a) = W2^e GELU(W1^e h_t(a))            e = 1..E                 (B, C, E, 2N+2+n_ev+1)
```
Split of `o` per (action, member): `vad ∈ R^N` (logit P(user active in slot s+1+k)), `onset ∈ R^{N+1}` (categorical
first user-active slot in the horizon, `N` = none), `overlap ∈ R` (P(user & AI both active in some horizon slot)),
`events ∈ R^{n_ev}` (P(listener event in horizon): barge-in, turn abandon, re-prompt), `cost ∈ R` (discounted
interaction cost-to-go `Σ_k γ^k c_{s+1+k}`, γ = 0.96, i.e. a ~1 s effective horizon). Uncertainty: epistemic =
std over the E members (trained on different bootstrap masks); aleatoric = mean Bernoulli entropy of `vad`.

**What it is (and is not).** Trained on logged data, `cost(a)` estimates `Q^{π_b}(h_t, a)` — the cost of taking `a` now and
then behaving like the logging policy `π_b`. This equals the *interventional* quantity only if the logged action was
not confounded by information the model cannot see (i.e. `a ⟂ unobserved listener state | history`). With randomised or
observation-only logging this holds; with human-to-human recordings it generally does not. Experiment E3 measures this.

## 5. Action selection (`ravel/policy.py`)

```
score_t(a) = Ê[cost|a] + w_ov P̂(overlap|a) + β std_e[cost|a] + κ compute(a) − α ρ_t 1[a is speech]
π_t(a)     = softmax(−score_t(a)/τ + λ log π_prior(a))            argmax at evaluation
```
`ρ_t = σ(MLP([y^A; M]))` = P(AI has relevant content now). `π_prior` = MLP on the core states, trained by behaviour
cloning and/or distillation of `softmax(−score/τ)` (stop-gradient). This is one-step greedy policy improvement
over the logging policy with an uncertainty penalty (cf. MOPO / PETS-style pessimism), *not* a new RL algorithm.

Decoding a behaviour-cloned prior (`select=`): `greedy` (argmax), `sample`, `hier` (argmax over action *modes*
first) and `thresh<θ>` (speak iff Σ_{speech actions} π(a) > θ, then argmax within the chosen group). Plain argmax of a
cloned turn-taking policy never starts speaking, because the logger spreads its onsets over time and over three
rates, so at every single slot "silent" stays the most probable action. The experiments therefore tune θ on
validation episodes for baseline B.

**Gradients.** RIF parameters receive gradients only from their supervised outcome losses; the RIF summary passed to the
planner is detached so planner/generator losses cannot distort the listener predictions. In `action_mode="gumbel"`
the policy prior is sampled with a straight-through Gumbel-softmax, so planner/DNAG losses reach the prior logits
(tested: `test_gumbel_mode_sends_gradient_into_policy`). The argmax at inference is non-differentiable by design.

## 6. D. Short-Horizon Acoustic Planner (`ravel/acoustic_planner.py`)

```
P_shift = [P_{1..K-1} ; p_empty]                                      (B, K=8, 512)
c       = MLP([y^A ; y^I ; e(a_{s+1}) ; sg(r)])   r = [P̂_vad(a)_{1..N} ; P̂(overlap|a) ; std_e cost(a)]
h       = PlanBlock^2(P_shift + pos + c, c)         self-attention over the 8 plan slots + FiLM(c)
g_k     = σ(w_g [P_shift,k ; h_k])                  per-slot revision gate
P       = (1−g) ⊙ P_shift + g ⊙ W_Δ h
commit  = P_0 → DNAG
```
Losses: DNAG CE on `P_0` (next slot), probe CE of codebook 0 on `P_k`, k ≥ 1 (slot `s+1+k`), activity BCE (is the AI
audible in slot `s+1+k`), and a small revision penalty `mean_k≥1 ||P_k − P_shift,k||²`.
The RIF → planner pathway is `r` (tested: `test_planner_shift_and_revision`).
*Honest caveat (from the prior-art review):* with an autoregressive token generator, uncommitted plan slots only matter
if something reads them. Here they feed the core (`mean_k P_k` → A state) and are trained to predict future audio, but
we have **not** shown that the editable plan improves anything over a direct slot head (`planner.enabled=false`).

## 7. E/F. Generator, codec and self-monitoring (`audio_generator.py`, `codec.py`, `feedback.py`)

DNAG: autoregressive GRU (d = 640) over the `L = F·Q = 12` tokens of a slot, frame-major, conditioned on `P_0` and on the
last emitted frame. Teacher forcing and sampling share the same factorisation (tested). Non-emitting actions
(`silent`, `yield`) are hard-gated to the codec's silence tokens at inference.

Codec: frozen EnCodec 24 kHz (causal, 75 Hz, 4 codebooks = 3 kbps, 14.85 M frozen parameters, not counted).
Streaming decode re-decodes the last 75 frames of context plus the 3 new frames (EnCodec's decoder contains an LSTM, so
this is approximate): measured SNR vs. whole-signal decoding = 17.9 / 24.9 / 27.6 dB for 3 / 15 / 75 context frames.

Self-monitoring feedback: `f_t = RMSNorm(MLP([Σ_q Emb(tok_{s,f,q})]_{f=1..3}) + E_act[a_s] + E_phase[t mod 2])`.
Separation of sources: (i) human speech — APF human-VAD head, input to H; (ii) own intended output — `f_t` into I and A;
(iii) echo — `ref_mel` input to the APF plus an optional DSP echo canceller (`inference/echo.py`, PBFDAF, ERLE > 20 dB on
a synthetic 75 ms echo path in tests; no double-talk detector).

## 8. Losses (`training/losses.py`)

| term | definition | weight (default) |
|---|---|---|
| dnag | CE of slot `s+1` tokens given `P_0` (all Q codebooks) | 1.0 |
| probe | CE of codebook 0, slots `s+2..s+K` from `P_1..P_{K-1}` | 0.2 |
| apc | L1(`ŷ_t`, mic_mel_{t+3}) | 0.5 |
| user_vad | BCE(p_user_t, human-VAD label) | 0.5 |
| rif_vad / onset / overlap / events | BCE / CE / BCE / BCE on the **logged** action's outcome, bootstrap-masked per member | 1 / .5 / .5 / .5 |
| rif_cost | Huber(δ=2) on discounted cost-to-go, masked if < 60 slots of future remain | 1.0 |
| bc | CE(π_prior, logged a_{s+1}) | 0.5 |
| distill | CE(π_prior, softmax(−score/τ)) | 0 (off) |
| relevance | BCE(ρ, has-relevant-content label) | 0.2 |
| plan_activity / plan_stability | BCE per plan slot / revision penalty | 0.3 / 0.01 |

## 9. Parameters and complexity (RAVEL-Tiny, measured with `RAVEL.param_breakdown()`)

| module | params |
|---|---|
| APF | 8.07 M |
| core (H, I, A, M) | 14.37 M |
| RIF | 6.16 M |
| policy | 0.86 M |
| planner | 8.94 M |
| DNAG | 11.07 M |
| feedback + action emb. | 1.72 M |
| **trainable total** | **51.21 M** |
| frozen EnCodec (excluded) | 14.85 M |

Per tick the core touches every core parameter once (~2 × 14.4 M ≈ 29 MFLOP) and the APF ~16 MFLOP; per slot the RIF
evaluates C candidates (~2·C·(~1.7 M) + 2·E·C·~0.25 M ≈ 40 MFLOP for C = 7), the planner ~2 × 9 M × 8 slots ≈ 0.14 GFLOP
(attention over 8 positions is negligible), and the DNAG 12 sequential GRU steps (~12 × 2 × 2.5 M ≈ 60 MFLOP).
Rough total ≈ 0.35 GFLOP per 40 ms slot ≈ 9 GFLOP/s of audio: tiny for a GPU, but the computation is a chain of small
sequential kernels, so latency, not FLOPs, is the constraint (see results).
