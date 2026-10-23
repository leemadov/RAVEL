# Prior-art review (focused) — RAVEL V2.1

Compiled 2026-10-23 from a web literature scan (arXiv, ACL Anthology, project pages, Google Patents).
It is a literature scan, not a freedom-to-operate opinion: "not found" means not found in a limited search.
Entries marked ✔ were re-checked against the primary source by reading the abstract or paper text. The other entries
come from the scan's search results and summaries and were not re-read line by line.

## Verdict first

**RAVEL is a new *combination* of known parts. None of its individual mechanisms is a new algorithm.**

| RAVEL mechanism | closest prior work | overlap | what might be new |
|---|---|---|---|
| Coupled recurrent H/A/I core, textless, 20 ms ticks | dGSLM (two-tower, cross-attention, two channels), Moshi (multi-stream), SpeechSSM (RG-LRU speech LM), Griffin/Hawk (RG-LRU), Roddy et al. 2018 (multiscale RNN for turn-taking) | substantial | Only the combination (pure gated diagonal recurrence, no attention, no text, full-duplex S2S). Whether it can match Transformer duplex quality is an open empirical question. |
| Reciprocal Influence Field + cost/uncertainty policy | ✔ Ekstedt et al. 2023 (VAP used as a *user model* on candidate synthesized speech), ✔ RC-TurnGPT (turn-taking conditioned on the candidate response), VAP (projection target), ✔ SyncLLM (forecasts the user's next chunk), PETS / MOPO (ensembles + uncertainty penalty), Khouzaimi et al. (Fitted-Q for micro-turn decisions) | substantial per component | Online scoring of a *discrete set of acoustic actions* (timing, backchannel, rate, emphasis) through an audio-level listener forecaster with an epistemic penalty, inside a full-duplex S2S model, was not found. The decision rule is textbook one-step greedy policy improvement / MPC. |
| Short-horizon editable acoustic plan (8 × 40 ms) | receding-horizon control (PETS, Diffusion Policy), sliding-window revision (AvatarForcing, rolling flow matching), CosyVoice 2 chunk look-ahead, Moshi acoustic delay | partial–substantial | Not found for speech codec tokens specifically. The mechanism itself is standard. **Caveat:** with an autoregressive token generator, uncommitted slots only matter if something reads them. |
| Codec generation, self-monitoring, echo/reference | Moshi (own stream as input), SALMONN-omni (listens to its own speech for echo robustness), LSLM, Qwen3-Omni | already exists | nothing material |

The key scientific risk for the RIF is **counterfactual identifiability, not architecture**. A model trained on logged
conversations only sees the actions the logger actually took. It is unbiased for untaken actions only when those
actions were not chosen using information the model cannot observe. RAVEL's experiment E3 tests exactly this point.

## Key sources

### Full-duplex / spoken dialogue models
* **Moshi** (Kyutai 2024), arXiv 2410.00037. Temporal Transformer (7B Helium) at 12.5 Hz over Mimi tokens, a Depth
  Transformer for the codebooks, parallel user/system audio streams and a time-aligned text "inner monologue". The loss
  also covers the user stream, so it learns a generative model of the user's audio, but that is not used to score its
  own candidate actions.
* ✔ **PersonaPlex** (NVIDIA, published 2026-01-15, research.nvidia.com/labs/adlr/personaplex): "built on the Moshi
  architecture from Kyutai, with 7 billion parameters", adds a voice prompt and a text role prompt; "a full duplex model:
  it listens and speaks at the same time". No new turn-taking mechanism relevant to RAVEL.
* **Qwen3-Omni** (Alibaba 2025), arXiv 2509.17765. A Thinker–Talker MoE with a streaming codec renderer. Duplex
  operation is not described in the paper (unverified).
* ✔ **SpeechSSM** (Park et al., arXiv 2412.18603): "Long-Form Speech Generation with Spoken Language Models", a textless,
  linear-time (state-space / recurrent) speech LM for multi-minute generation. The scan reports a Griffin-style hybrid
  with RG-LRU blocks; that detail was not re-verified from the abstract. Single-speaker continuation, not dialogue.
* **dGSLM** (Nguyen et al., TACL 2023), arXiv 2203.16502. Two-channel textless dialogue with a dual-tower Transformer and
  cross-attention, trained on Fisher. It generates overlap and backchannels.
* ✔ **SyncLLM** (Veluri et al., EMNLP 2024), arXiv 2409.15594: "Beyond Turn-Based Interfaces: Synchronous LLMs as
  Full-Duplex Dialogue Agents". Llama3-8B is made time-synchronous, and it forecasts the user's next chunk to mask
  latency of up to 240 ms.
* Also scanned: OmniFlatten (2410.17799), LSLM (2408.02622), Freeze-Omni (2411.00774), MinMo (2501.06282), SALMONN-omni
  (2411.18138), DuplexMamba (2502.11123, Mamba-based duplex but speech-to-text), NTPP (2506.00975), Sesame CSM (not
  full-duplex), LFM2-Audio (duplex status unverified).

### Turn-taking prediction and listener models
* **VAP** (Ekstedt & Skantze 2022), arXiv 2205.09812. Predicts both speakers' future voice activity in discrete bins
  over 2 s from stereo audio. RIF's per-slot user-VAD head is essentially a VAP-style projection target.
* ✔ **Ekstedt, Wang, Székely, Gustafson, Skantze — "Automatic Evaluation of Turn-taking Cues in Conversational Speech
  Synthesis"** (Interspeech 2023, arXiv 2305.17971). Quote: "we use the model to assess the likelihood that a user would
  take the turn at each frame in a synthesized spoken utterance … In this way, the VAP model works as a user model."
  They do this offline and suggest it as a possible TTS training objective. **This is the closest prior art to the RIF.**
  It predicts listener turn-taking conditioned on the system's own candidate audio. It has no action set, no online
  action selection, no cost model and no uncertainty estimate.
* ✔ **RC-TurnGPT** (Jiang, Ekstedt, Skantze, Findings of ACL 2023, arXiv 2305.02036): turn-end prediction conditioned on
  "what the next speaker intends to say" (text only).
* Roddy & Harte 2020 (arXiv 2005.09128): response-timing distributions conditioned on the system's response.
* RL for turn-taking: Jonsdottir et al. 2008; Khouzaimi et al. (SIGDIAL 2015, IJCAI 2016; Fitted-Q with a user
  simulator). DuplexPO (arXiv 2607.07148, GRPO on a hand-designed turn-taking reward) was found in the scan but not
  re-verified.
* Model-based / offline RL machinery reused by RAVEL's policy: PETS (1805.12114, probabilistic ensembles + MPC), MOPO
  (2005.13239, uncertainty-penalised offline model-based RL).

### Receding-horizon / revisable streaming generation
Incremental TTS with look-ahead (1911.02750), CosyVoice 2 chunk-aware streaming flow matching (2412.10117), Kyutai DSM
delayed streams (2509.08753), and sliding-window revision in audio-driven video (AvatarForcing 2603.14331, RFLAV
2503.08307). No speech generator with an explicit multi-slot editable future buffer that commits one slot per step was
found.

### Patents (Google Patents, not exhaustive)
US20080004881A1 "Turn-taking model" (2005); WO2011119168A1 (Nuance, barge-in probability conditioned on the system
prompt's transition-relevance places); US20200184956A1 (Microsoft, predicts the user's response to a system prompt; about
content, not timing); EP3714453B1 (Microsoft, full-duplex chatbot); US6941269B1 (AT&T, rule-based backchannels).
No patent on a neural action-conditioned listener forecaster used for action selection was found. A professional search
would be needed before any IP claim.

## Design consequences adopted in V2.1
1. The RIF is framed as an *action-conditioned predictor* (Q^{π_b} plus listener outcomes), not a causal model. Its
   counterfactual accuracy is measured with simulator interventions (E2), and the effect of confounded logs is
   measured (E3).
2. The planner is reported as a supporting mechanism with an explicit ablation flag (`planner.enabled=false`). No
   benefit is claimed without an experiment.
3. Recommended redesign for a future version, **not implemented**: let the RIF score *planned audio* (the plan's
   predicted activity/prosody over the horizon) instead of only a discrete action label. That would make the editable
   plan matter and is the VAP-as-user-model idea moved online.
