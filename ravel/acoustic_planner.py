"""D. Short-Horizon Acoustic Planner (SHAP).

Plan P_t in R^{K x d}: K = 8 editable slots of 40 ms (320 ms). Updated once per slot decision:

    P_shift  = Shift(P_{t-1})            drop slot 0 (it was committed and emitted), append learned empty slot
    c        = MLP([y_A ; y_I ; e(a_t) ; r_t])   r_t = RIF summary for the selected action
                                                 (predicted user VAD over the horizon, P(overlap), std)
    h        = PlanBlocks(P_shift + pos, c)      attention *within the plan* (8 positions) + FiLM(c)
    g_k      = sigmoid(w_g [P_shift_k ; h_k])    per-slot revision gate
    P_t      = (1 - g) * P_shift + g * W_d h     revise uncommitted slots
    commit   = P_t[:, 0]  -> DNAG -> codec tokens for the next 40 ms

Causal validity: audio already emitted is never in the plan (it was shifted out), so it cannot be
revised; the only path from new audio to output is through slots that have not been emitted yet.
The RIF summary enters through c, so a predicted user onset can push upcoming slots toward silence.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PlanBlock(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.mod = nn.Linear(d, 4 * d)  # FiLM (scale, shift) for both sub-layers
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        s1, b1, s2, b2 = self.mod(c)[:, None, :].chunk(4, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x) * (1 + s2) + b2
        return x + self.ffn(h)


class ShortHorizonPlanner(nn.Module):
    def __init__(self, n_slots: int, d_slot: int, n_layers: int, n_heads: int, d_a: int, d_i: int, d_act: int,
                 d_rif: int, probe_out: int):
        super().__init__()
        self.K, self.d = n_slots, d_slot
        self.empty = nn.Parameter(torch.randn(d_slot) * 0.02)
        self.pos = nn.Parameter(torch.randn(n_slots, d_slot) * 0.02)
        self.cond = nn.Sequential(nn.Linear(d_a + d_i + d_act + d_rif, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot))
        self.blocks = nn.ModuleList([PlanBlock(d_slot, n_heads) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_slot)
        self.delta = nn.Linear(d_slot, d_slot)
        self.gate = nn.Linear(2 * d_slot, 1)
        self.activity = nn.Linear(d_slot, 1)  # P(AI audible in that future slot)
        self.probe = nn.Linear(d_slot, probe_out) if probe_out > 0 else None
        self.d_rif = d_rif

    def init_plan(self, batch: int) -> torch.Tensor:
        return self.empty.expand(batch, self.K, self.d).clone()

    def shift(self, P: torch.Tensor) -> torch.Tensor:
        return torch.cat([P[:, 1:], self.empty.expand(P.shape[0], 1, self.d)], dim=1)

    def forward(self, P_prev, y_a, y_i, act_e, rif_sum=None):
        B = P_prev.shape[0]
        Ps = self.shift(P_prev)
        if self.d_rif and rif_sum is None:
            rif_sum = P_prev.new_zeros(B, self.d_rif)
        cin = torch.cat([y_a, y_i, act_e] + ([rif_sum] if self.d_rif else []), -1)
        c = self.cond(cin)
        h = Ps + self.pos + c[:, None, :]
        for blk in self.blocks:
            h = blk(h, c)
        h = self.norm(h)
        g = torch.sigmoid(self.gate(torch.cat([Ps, h], -1)))  # (B, K, 1)
        P = (1 - g) * Ps + g * self.delta(h)
        return P, {"gate": g.squeeze(-1), "revision": (P - Ps).pow(2).mean(-1), "shifted": Ps}


class DirectSlotHead(nn.Module):
    """Ablation: no editable plan; the next slot vector is produced directly from the current state."""

    def __init__(self, d_slot: int, d_a: int, d_i: int, d_act: int, d_rif: int):
        super().__init__()
        self.K, self.d, self.d_rif = 1, d_slot, d_rif
        self.net = nn.Sequential(nn.Linear(d_a + d_i + d_act + d_rif, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot))
        self.activity = nn.Linear(d_slot, 1)
        self.probe = None

    def init_plan(self, batch: int) -> torch.Tensor:
        return self.activity.weight.new_zeros(batch, 1, self.d)

    def forward(self, P_prev, y_a, y_i, act_e, rif_sum=None):
        B = y_a.shape[0]
        if self.d_rif and rif_sum is None:
            rif_sum = y_a.new_zeros(B, self.d_rif)
        P = self.net(torch.cat([y_a, y_i, act_e] + ([rif_sum] if self.d_rif else []), -1))[:, None, :]
        z = P.new_zeros(B, 1)
        return P, {"gate": z + 1, "revision": z, "shifted": P_prev}


def plan_probe_logits(planner, P: torch.Tensor, frames: int, codebooks: int, vocab: int):
    """Cheap multi-horizon readout of plan slots k>=1: logits (B, K-1, frames, codebooks, vocab)."""
    if planner.probe is None:
        return None
    B, K, _ = P.shape
    return planner.probe(P[:, 1:]).view(B, K - 1, frames, codebooks, vocab)
