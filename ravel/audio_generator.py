"""E. Direct Neural Audio Generator (DNAG).

Maps the committed plan slot (one 40 ms slot = F=3 EnCodec frames x Q codebooks) to codec tokens,
autoregressively over the L = F*Q positions in frame-major order j = f*Q + q:

    x_j   = Emb(token_{j-1})  (BOS at j=0)  + pos_j + W_c slot
    h_0   = tanh(W_h [W_c slot + sum_q Emb_prev(prev_frame_q)])     continuity with the last emitted frame
    h_j   = GRU(x_j, h_{j-1})
    p(token_j) = softmax(W_{q(j)} RMSNorm(h_j))

The frozen codec decoder turns tokens into a 24 kHz waveform. No text, no TTS pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectNeuralAudioGenerator(nn.Module):
    def __init__(self, d_slot: int, d_model: int, n_q: int, vocab: int, frames_per_slot: int):
        super().__init__()
        self.F, self.Q, self.V, self.d = frames_per_slot, n_q, vocab, d_model
        self.L = frames_per_slot * n_q
        self.bos = n_q * vocab
        self.tok_emb = nn.Embedding(n_q * vocab + 1, d_model)
        self.prev_emb = nn.Embedding(n_q * vocab, d_model)
        self.pos = nn.Parameter(torch.randn(self.L, d_model) * 0.02)
        self.cond = nn.Linear(d_slot, d_model)
        self.h0 = nn.Linear(d_model, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.out_norm = nn.RMSNorm(d_model)
        self.heads = nn.Parameter(torch.randn(n_q, d_model, vocab) * d_model ** -0.5)
        self.register_buffer("q_off", torch.arange(n_q) * vocab, persistent=False)

    def _prep(self, slot, prev_frame):
        c = self.cond(slot)  # (N, d)
        prev = self.prev_emb(prev_frame + self.q_off).sum(1)  # (N, d)
        h0 = torch.tanh(self.h0(c + prev))[None]  # (1, N, d)
        return c, h0

    def _logits(self, h, q_idx):
        """h (N, L', d) at positions whose codebooks are q_idx (L',) -> (N, L', V)."""
        h = self.out_norm(h)
        return torch.einsum("nld,ldv->nlv", h, self.heads[q_idx])

    def forward(self, slot: torch.Tensor, prev_frame: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Teacher forcing. slot (N, d_slot), prev_frame (N, Q), tgt (N, F, Q) -> logits (N, F, Q, V)."""
        N = slot.shape[0]
        c, h0 = self._prep(slot, prev_frame)
        flat = tgt.reshape(N, self.L) + self.q_off.repeat(self.F)  # token ids with codebook offsets
        inp = torch.cat([flat.new_full((N, 1), self.bos), flat[:, :-1]], 1)
        x = self.tok_emb(inp) + self.pos + c[:, None, :]
        h, _ = self.gru(x, h0)
        q_idx = torch.arange(self.Q, device=slot.device).repeat(self.F)
        return self._logits(h, q_idx).view(N, self.F, self.Q, self.V)

    @torch.no_grad()
    def generate(self, slot: torch.Tensor, prev_frame: torch.Tensor, temperature: float = 1.0,
                 top_k: int | None = None, greedy: bool = False) -> torch.Tensor:
        """slot (N, d_slot), prev_frame (N, Q) -> tokens (N, F, Q)."""
        N = slot.shape[0]
        c, h = self._prep(slot, prev_frame)
        tok = slot.new_full((N,), self.bos, dtype=torch.long)
        out = []
        for j in range(self.L):
            q = j % self.Q
            x = self.tok_emb(tok) + self.pos[j] + c
            o, h = self.gru(x[:, None, :], h)
            logits = self._logits(o, torch.tensor([q], device=slot.device))[:, 0]
            if greedy or temperature <= 0:
                v = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_k:
                    kth = logits.topk(top_k, -1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                v = torch.distributions.Categorical(logits=logits).sample()
            out.append(v)
            tok = v + q * self.V
        return torch.stack(out, 1).view(N, self.F, self.Q)


def token_ce(logits: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean CE over tokens. logits (..., V), tgt (...), mask broadcastable to tgt."""
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1), reduction="none")
    ce = ce.view(tgt.shape)
    if mask is None:
        return ce.mean()
    mask = mask.expand_as(ce).float()
    return (ce * mask).sum() / mask.sum().clamp(min=1.0)
