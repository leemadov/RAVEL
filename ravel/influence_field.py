"""C. Reciprocal Influence Field (RIF).

An action-conditioned predictor of the *listener's* near-future observable behaviour:

    c_t        = MLP([y_H ; y_A ; y_I ; M])                                (B, d)
    h_t(a)     = c_t * (1 + gamma(e_a)) + beta(e_a)                        FiLM by candidate action, (B, C, d)
    h_t(a)     = ResMLP^L(h_t(a))
    o_t^e(a)   = W2^e GELU(W1^e h_t(a))   for ensemble members e = 1..E    (B, C, E, n_out)

o is split into (per candidate action a, per ensemble member e):
    vad      (N)    logits of P(user voice-active in future slot k), k = 1..N
    onset    (N+1)  logits of the slot index of the user's first voice activity in the horizon (N = none)
    overlap  (1)    logit of P(user and AI both active in some slot of the horizon)
    events   (n_ev) logits of discrete listener events in the horizon (e.g. barge-in, turn abandon, re-prompt)
    cost     (1)    discounted interaction cost-to-go  sum_k gamma^k c_{t+k}  ("appropriateness" = -cost)

Uncertainty: epistemic = std over the E members (bootstrapped heads); aleatoric = mean Bernoulli
entropy of the VAD predictions.

IMPORTANT: trained on logged conversations this is an *observational* conditional model
p(listener | history, a_t = a). It only estimates the effect of *intervening* with a if the logged
action was not confounded with unobserved listener state (see docs/SPEC.md, "Causality").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLP(nn.Module):
    def __init__(self, d: int, mult: int = 2):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.fc1, self.fc2 = nn.Linear(d, mult * d), nn.Linear(mult * d, d)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class ReciprocalInfluenceField(nn.Module):
    def __init__(self, d_h: int, d_a: int, d_i: int, d_m: int, d_act: int, d_hidden: int, d_head: int,
                 n_layers: int, horizon: int, n_ensemble: int, n_events: int, listener_heads: bool = True):
        super().__init__()
        self.N, self.E, self.n_events, self.listener_heads = horizon, n_ensemble, n_events, listener_heads
        self.ctx = nn.Sequential(nn.Linear(d_h + d_a + d_i + d_m, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden))
        self.film = nn.Linear(d_act, 2 * d_hidden)
        self.trunk = nn.ModuleList([ResMLP(d_hidden) for _ in range(n_layers)])
        self.out_norm = nn.RMSNorm(d_hidden)
        N = horizon
        self.split = [N, N + 1, 1, n_events, 1] if listener_heads else [1]
        n_out = sum(self.split)
        self.w1 = nn.Parameter(torch.randn(n_ensemble, d_hidden, d_head) * d_hidden ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(n_ensemble, d_head))
        self.w2 = nn.Parameter(torch.randn(n_ensemble, d_head, n_out) * d_head ** -0.5 * 0.5)
        self.b2 = nn.Parameter(torch.zeros(n_ensemble, n_out))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, y_h, y_a, y_i, m, act_tab: torch.Tensor) -> dict:
        """y_* (B, d_*); act_tab (C, d_act) for all candidates or (B, C, d_act).  Returns dict of (B, C, E, ...)."""
        c = self.ctx(torch.cat([y_h, y_a, y_i, m], -1))  # (B, d)
        gamma, beta = self.film(act_tab).chunk(2, -1)
        if gamma.dim() == 2:
            gamma, beta = gamma[None], beta[None]
        h = c[:, None, :] * (1.0 + gamma) + beta  # (B, C, d)
        for blk in self.trunk:
            h = blk(h)
        h = self.out_norm(h)
        e = F.gelu(torch.einsum("bcd,edk->bcek", h, self.w1) + self.b1)
        o = torch.einsum("bcek,eko->bceo", e, self.w2) + self.b2  # (B, C, E, n_out)
        if not self.listener_heads:
            return {"cost": o[..., 0]}
        vad, onset, overlap, events, cost = o.split(self.split, dim=-1)
        return {"vad": vad, "onset": onset, "overlap": overlap[..., 0], "events": events, "cost": cost[..., 0]}

    @staticmethod
    def summarize(out: dict) -> dict:
        """Ensemble-averaged predictions and uncertainty, each (B, C, ...)."""
        s = {"cost_mean": out["cost"].mean(2), "cost_std": out["cost"].std(2, unbiased=False)}
        if "vad" in out:
            p_vad = torch.sigmoid(out["vad"]).mean(2)  # (B, C, N)
            s["p_vad"] = p_vad
            s["p_overlap"] = torch.sigmoid(out["overlap"]).mean(2)
            s["p_events"] = torch.sigmoid(out["events"]).mean(2)
            s["p_onset"] = torch.softmax(out["onset"], -1).mean(2)  # (B, C, N+1)
            pv = p_vad.clamp(1e-6, 1 - 1e-6)
            s["aleatoric"] = -(pv * pv.log() + (1 - pv) * (1 - pv).log()).mean(-1)
        else:
            s["p_overlap"] = torch.zeros_like(s["cost_mean"])
        return s
