"""Action selection.

RIF mode (full RAVEL):
    score_t(a) = E[cost | a] + w_ov * P(overlap | a) + beta * std_e[cost | a]
                 + kappa * compute(a) - alpha * rho_t * 1[a is speech]
    pi_t(a)    = softmax(-score_t(a) / tau + lambda * log pi_prior(a))
Prior-only mode (RAVEL without RIF, baseline B):
    pi_t(a)    = pi_prior(a)  =  softmax(MLP([y_H ; y_A ; y_I ; M]))

rho_t = P(AI has relevant content to deliver now), a learned head on (A, M).
Selection is argmax at evaluation; during joint training a straight-through Gumbel-softmax sample
lets planner/generator losses send gradient into the policy logits. The RIF itself is trained only
by its supervised outcome losses (its scores are detached when distilled into the prior).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actions import ActionSpace


class InteractionPolicy(nn.Module):
    def __init__(self, d_h: int, d_a: int, d_i: int, d_m: int, d_hidden: int, space: ActionSpace, pcfg):
        super().__init__()
        self.pcfg = pcfg
        C = len(space)
        self.prior = nn.Sequential(nn.Linear(d_h + d_a + d_i + d_m, d_hidden), nn.GELU(), nn.Linear(d_hidden, C))
        self.relevance = nn.Sequential(nn.Linear(d_a + d_m, d_hidden // 2), nn.GELU(), nn.Linear(d_hidden // 2, 1))
        self.register_buffer("compute_cost", space.compute_cost.clone(), persistent=False)
        self.register_buffer("is_speech", space.is_speech.float(), persistent=False)

    def prior_logits(self, y_h, y_a, y_i, m) -> torch.Tensor:
        return self.prior(torch.cat([y_h, y_a, y_i, m], -1))

    def relevance_logit(self, y_a, m) -> torch.Tensor:
        return self.relevance(torch.cat([y_a, m], -1)).squeeze(-1)

    def score(self, summary: dict, rho: torch.Tensor) -> torch.Tensor:
        p = self.pcfg
        return (summary["cost_mean"] + p.w_overlap * summary["p_overlap"] + p.beta_uncert * summary["cost_std"]
                + p.kappa_compute * self.compute_cost - p.alpha_relevance * rho[..., None] * self.is_speech)

    def logits(self, score: torch.Tensor | None, prior_logits: torch.Tensor) -> torch.Tensor:
        if score is None:
            return prior_logits
        p = self.pcfg
        return -score / p.temperature + p.prior_weight * F.log_softmax(prior_logits, -1)

    @staticmethod
    def select(logits: torch.Tensor, mode: str = "greedy", tau: float = 1.0):
        """Returns (index (B,), one-hot-like weights (B, C)); 'gumbel' is straight-through."""
        if mode == "greedy":
            idx = logits.argmax(-1)
            return idx, F.one_hot(idx, logits.shape[-1]).to(logits.dtype)
        if mode == "sample":
            idx = torch.distributions.Categorical(logits=logits).sample()
            return idx, F.one_hot(idx, logits.shape[-1]).to(logits.dtype)
        if mode == "gumbel":
            y = F.gumbel_softmax(logits, tau=tau, hard=True)
            return y.argmax(-1), y
        raise ValueError(mode)
