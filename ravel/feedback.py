"""F. Self-Monitoring Feedback (SMF).

The AI always knows exactly what it is emitting: the tokens of slot s are committed at the end of
slot s-1. During the two ticks of slot s the core receives

    f_t = RMSNorm( MLP( [sum_q Emb(tok_{s,f,q})]_{f=1..F} ) + E_act[a_s] + E_phase[t mod 2] )

so the AI's own speaking history is an explicit, separately-labelled input. Together with the
loudspeaker reference given to the APF (ref_mel) this lets the model separate
  * human microphone speech        (APF human-VAD head, H state)
  * its own intended output        (f_t -> I and A states)
  * echo / playback leakage        (mic energy explained by the reference)
and is what prevents its own voice from being treated as new human input.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SelfMonitoringFeedback(nn.Module):
    def __init__(self, n_q: int, vocab: int, frames_per_slot: int, d_f: int, n_actions: int, ticks_per_slot: int):
        super().__init__()
        self.Q, self.V, self.F, self.tps = n_q, vocab, frames_per_slot, ticks_per_slot
        self.tok = nn.Embedding(n_q * vocab, d_f)
        self.mix = nn.Sequential(nn.Linear(frames_per_slot * d_f, d_f), nn.GELU(), nn.Linear(d_f, d_f))
        self.act = nn.Embedding(n_actions, d_f)
        self.phase = nn.Embedding(ticks_per_slot, d_f)
        self.norm = nn.RMSNorm(d_f)
        self.register_buffer("q_off", torch.arange(n_q) * vocab, persistent=False)

    def slot_embed(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """tokens (..., F, Q), actions (...) -> (..., d_f)."""
        e = self.tok(tokens + self.q_off).sum(-2)  # (..., F, d_f)
        return self.mix(e.flatten(-2)) + self.act(actions)

    def forward(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """tokens (B, S, F, Q), actions (B, S) -> per-tick feedback (B, S*tps, d_f)."""
        e = self.slot_embed(tokens, actions)  # (B, S, d_f)
        e = e[:, :, None, :] + self.phase.weight[None, None]  # (B, S, tps, d_f)
        return self.norm(e.flatten(1, 2))

    def step(self, slot_tokens: torch.Tensor, action: torch.Tensor, phase: int) -> torch.Tensor:
        """Streaming: slot_tokens (B, F, Q), action (B,) -> (B, d_f)."""
        return self.norm(self.slot_embed(slot_tokens, action) + self.phase.weight[phase])
