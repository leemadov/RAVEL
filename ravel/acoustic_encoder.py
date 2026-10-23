"""A. Acoustic Perception Field (APF).

Causal, streaming encoder: per 20 ms tick it receives the log-mel of the microphone and the log-mel
of the AI's own output reference (what the loudspeaker is playing), and produces z_t.

    u_t   = W_in [norm(mic_mel_t) ; norm(ref_mel_t)]                  (B, d)
    c_t   = GLU-CausalConv stack(u_{<=t})                             receptive field 1 + (k-1)*sum(dilations) ticks
    g_t   = GRU(c_t, g_{t-1})                                          fast acoustic memory
    z_t   = RMSNorm(c_t + g_t)                                         (B, d_z)

Auxiliary heads: p(user speaking | mic, ref)  (echo-robust human VAD) and APC future-mel prediction.
Reference conditioning lets the encoder learn to discount playback leakage of the AI's own voice.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MEL_MEAN, MEL_STD = -6.0, 4.0  # fixed input normalisation for natural-log mel power of [-1,1] audio


class CausalConvBlock(nn.Module):
    def __init__(self, d: int, kernel: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.conv = nn.Conv1d(d, 2 * d, kernel, dilation=dilation)
        self.pad = (kernel - 1) * dilation
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """x (B, T, d); state (B, d, pad) holds the previous normalised inputs."""
        B, T, d = x.shape
        h = self.norm(x).transpose(1, 2)
        if state is None:
            state = h.new_zeros(B, d, self.pad)
        h = torch.cat([state, h], dim=-1)
        new_state = h[..., h.shape[-1] - self.pad:]
        y = F.glu(self.conv(h), dim=1).transpose(1, 2)
        return x + self.drop(y), new_state


class AcousticPerceptionField(nn.Module):
    def __init__(self, n_mels: int, d_model: int, kernel: int, dilations: list[int], dropout: float = 0.0):
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.in_proj = nn.Linear(2 * n_mels, d_model)
        self.blocks = nn.ModuleList([CausalConvBlock(d_model, kernel, d, dropout) for d in dilations])
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.out_norm = nn.RMSNorm(d_model)
        self.user_vad_head = nn.Linear(d_model, 1)
        self.apc_head = nn.Linear(d_model, n_mels)
        self.receptive_field = 1 + sum((kernel - 1) * d for d in dilations)

    def init_state(self, batch: int, device=None) -> dict:
        p = next(self.parameters())
        return {
            "conv": [p.new_zeros(batch, self.d_model, b.pad) for b in self.blocks],
            "gru": p.new_zeros(1, batch, self.d_model),
        }

    def forward(self, mic_mel: torch.Tensor, ref_mel: torch.Tensor, state: dict | None = None):
        """mic_mel, ref_mel (B, T, n_mels) -> z (B, T, d), new_state, aux dict."""
        x = torch.cat([(mic_mel - MEL_MEAN) / MEL_STD, (ref_mel - MEL_MEAN) / MEL_STD], dim=-1)
        h = self.in_proj(x)
        conv_states = state["conv"] if state is not None else [None] * len(self.blocks)
        new_conv = []
        for blk, st in zip(self.blocks, conv_states):
            h, ns = blk(h, st)
            new_conv.append(ns)
        g, g_state = self.gru(h, state["gru"] if state is not None else None)
        z = self.out_norm(h + g)
        aux = {
            "user_vad_logit": self.user_vad_head(z).squeeze(-1),  # (B, T)
            "apc_pred": self.apc_head(z),                          # (B, T, n_mels) -> mel at t+offset
        }
        return z, {"conv": new_conv, "gru": g_state}, aux
