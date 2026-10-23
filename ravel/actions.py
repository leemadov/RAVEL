"""Acoustic action space.

An action governs what the AI emits during the *next* 40 ms slot. Actions are factorised into
mode / rate / emphasis so that the influence field and planner see a structured embedding
rather than an opaque index.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# name -> (mode, rate, emphasis, emits_speech, compute_cost)
#   mode: 0 silent, 1 yield, 2 backchannel, 3 speak
#   rate: 0 slow, 1 normal, 2 fast
#   emphasis: 0 normal, 1 high
ACTION_TABLE = {
    "silent": (0, 1, 0, False, 0.0),
    "yield": (1, 1, 0, False, 0.0),
    "backchannel": (2, 1, 0, True, 0.3),
    "speak": (3, 1, 0, True, 1.0),
    "speak_slow": (3, 0, 0, True, 1.0),
    "speak_fast": (3, 2, 0, True, 1.0),
    "speak_emph": (3, 1, 1, True, 1.0),
}


class ActionSpace:
    def __init__(self, names: list[str]):
        unknown = [n for n in names if n not in ACTION_TABLE]
        if unknown:
            raise ValueError(f"Unknown actions {unknown}; known: {list(ACTION_TABLE)}")
        self.names = list(names)
        rows = [ACTION_TABLE[n] for n in names]
        self.mode = torch.tensor([r[0] for r in rows])
        self.rate = torch.tensor([r[1] for r in rows])
        self.emph = torch.tensor([r[2] for r in rows])
        self.emits = torch.tensor([r[3] for r in rows])
        self.compute_cost = torch.tensor([r[4] for r in rows], dtype=torch.float32)
        # "speech-like" actions = full speech (not backchannel); used for relevance bonus
        self.is_speech = torch.tensor([r[0] == 3 for r in rows])

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)


class ActionEmbedding(nn.Module):
    """Factorised embedding e(a) = E_mode[m] + E_rate[r] + E_emph[e]  ->  (d_act,)."""

    def __init__(self, space: ActionSpace, d_act: int):
        super().__init__()
        self.space = space
        self.mode = nn.Embedding(4, d_act)
        self.rate = nn.Embedding(3, d_act)
        self.emph = nn.Embedding(2, d_act)
        self.register_buffer("_mode", space.mode.clone(), persistent=False)
        self.register_buffer("_rate", space.rate.clone(), persistent=False)
        self.register_buffer("_emph", space.emph.clone(), persistent=False)

    def table(self) -> torch.Tensor:
        """(C, d_act) embeddings of every candidate action."""
        return self.mode(self._mode) + self.rate(self._rate) + self.emph(self._emph)

    def forward(self, idx_or_probs: torch.Tensor) -> torch.Tensor:
        """Integer indices (...,) -> (..., d_act); or probabilities (..., C) -> soft embedding."""
        tab = self.table()
        if idx_or_probs.dtype in (torch.long, torch.int32, torch.int64):
            return tab[idx_or_probs]
        return idx_or_probs @ tab
