"""Baseline A — conventional streaming autoregressive codec-token model.

No coupled states, no influence field, no editable plan. A frame-rate (75 Hz) stacked LSTM reads
[sum_q Emb(tok_{i,q}) + W (mic_mel, ref_mel at the latest complete 20 ms tick)] and a depth GRU
predicts the Q codebooks of frame i+1 (same depth-decoder class as RAVEL's DNAG with F=1).
For duplex experiments a policy head reads the LSTM state at slot boundaries.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .audio_generator import DirectNeuralAudioGenerator
from .acoustic_encoder import MEL_MEAN, MEL_STD


def frame_to_tick_index(n_frames: int, hop: int = 320, tick: int = 480) -> torch.Tensor:
    """Latest complete tick at the end of each codec frame (-1 if none)."""
    ends = (torch.arange(n_frames) + 1) * hop
    return torch.div(ends, tick, rounding_mode="floor") - 1


class StreamingARBaseline(nn.Module):
    def __init__(self, n_mels: int, n_q: int, vocab: int, d_model: int, n_layers: int, d_depth: int,
                 n_actions: int, frames_per_slot: int = 3):
        super().__init__()
        self.Q, self.V, self.Fs = n_q, vocab, frames_per_slot
        self.mic_proj = nn.Linear(2 * n_mels, d_model)
        self.tok = nn.Embedding(n_q * vocab, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True)
        self.depth = DirectNeuralAudioGenerator(d_model, d_depth, n_q, vocab, 1)
        self.policy = nn.Linear(d_model, n_actions)
        self.register_buffer("q_off", torch.arange(n_q) * vocab, persistent=False)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, mic_mel: torch.Tensor, ref_mel: torch.Tensor, tokens: torch.Tensor):
        """mic/ref_mel (B, T, n_mels) at 50 Hz; tokens (B, Tf, Q) at 75 Hz.
        Returns h (B, Tf, d): state after reading frame i (predicts frame i+1)."""
        B, Tf, Q = tokens.shape
        idx = frame_to_tick_index(Tf).to(tokens.device)
        feats = torch.cat([(mic_mel - MEL_MEAN) / MEL_STD, (ref_mel - MEL_MEAN) / MEL_STD], -1)
        fa = feats[:, idx.clamp(min=0)] * (idx >= 0).float()[None, :, None]
        x = self.tok(tokens + self.q_off).sum(2) + self.mic_proj(fa)
        h, _ = self.lstm(x)
        return h

    def token_logits(self, h: torch.Tensor, tokens: torch.Tensor, select: torch.Tensor | None = None):
        """Teacher-forced logits for frames 1..Tf-1: returns (N, 1, Q, V), targets (N, 1, Q)."""
        hh, prev, tgt = h[:, :-1], tokens[:, :-1], tokens[:, 1:]
        if select is not None:
            hh, prev, tgt = hh[select], prev[select], tgt[select]
        else:
            hh, prev, tgt = hh.flatten(0, 1), prev.flatten(0, 1), tgt.flatten(0, 1)
        tgt = tgt[:, None]
        return self.depth(hh, prev, tgt), tgt

    def slot_states(self, h: torch.Tensor) -> torch.Tensor:
        """State at the end of each slot (after its last frame): (B, S, d)."""
        return h[:, self.Fs - 1::self.Fs]
