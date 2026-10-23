"""B. Dual-State Conversational Core (DSCC).

Three coupled recurrent states updated every 20 ms tick, plus a clocked slow memory:

  H  (d_h)  human state       driven by perception z_t and the human-VAD estimate
  I  (d_i)  interaction state driven by H_t, A_{t-1}, own-output feedback f_t and the current action
  A  (d_a)  AI state          driven by H_t, I_t, the current plan summary, slow memory and f_t
  M  (d_m)  slow conversational memory, GRU updated every `slow_every` ticks (200 ms) from the
            window-average of [y_H, y_A, y_I]   ("clockwork" slow path)

Each fast state uses a gated linear recurrence in the style of the RG-LRU (Griffin, De et al. 2024):

  [x, r, i] = W u_t
  a_t       = sigma(Lambda) ** (c * sigmoid(r))         per-channel, input-dependent decay in (0, 1)
  s_t       = a_t * s_{t-1} + sqrt(1 - a_t^2) * (sigmoid(i) * x)
  y_t       = s_t + FFN(RMSNorm(s_t))

Lambda is initialised so that a^c spans a per-state range of time constants (I fast, A slow), which
separates fast acoustic/interaction memory from slower conversational memory. The coupling between
states makes the core a non-linear RNN, so it is stepped tick by tick (O(T) sequential, O(1) memory
per step at inference).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLRUCell(nn.Module):
    def __init__(self, d_in: int, d: int, c: float = 8.0, a_range=(0.9, 0.999), ffn_mult: int = 3):
        super().__init__()
        self.d, self.c = d, c
        self.in_proj = nn.Linear(d_in, 3 * d)
        a0 = torch.empty(d).uniform_(*a_range)
        self.lam = nn.Parameter(torch.logit(a0 ** (1.0 / c)))
        self.norm = nn.RMSNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, u: torch.Tensor, s_prev: torch.Tensor):
        x, r, i = self.in_proj(u).chunk(3, dim=-1)
        log_a = -self.c * torch.sigmoid(r) * F.softplus(-self.lam)
        a = torch.exp(log_a)
        s = a * s_prev + torch.sqrt(torch.clamp(1.0 - a * a, min=1e-6)) * (torch.sigmoid(i) * x)
        y = s + self.ffn(self.norm(s))
        return s, y

    def time_constants_ticks(self) -> torch.Tensor:
        """Time constant 1/(1-a) of each channel at full recurrence gate (r -> +inf)."""
        a = torch.exp(-self.c * F.softplus(-self.lam))
        return 1.0 / (1.0 - a)


class DualStateCore(nn.Module):
    def __init__(self, d_z: int, d_f: int, d_act: int, d_plan: int, d_h: int, d_a: int, d_i: int,
                 d_m: int, d_couple: int, slow_every: int, ffn_mult: int = 3, lru_c: float = 8.0):
        super().__init__()
        dc = d_couple
        self.dims = dict(h=d_h, a=d_a, i=d_i, m=d_m)
        self.slow_every = slow_every
        # coupling projections (state -> small coupling vector)
        self.a2h, self.i2h, self.m2h = nn.Linear(d_a, dc), nn.Linear(d_i, dc), nn.Linear(d_m, dc)
        self.h2i, self.a2i = nn.Linear(d_h, dc), nn.Linear(d_a, dc)
        self.h2a, self.i2a, self.p2a, self.m2a = (nn.Linear(d_h, dc), nn.Linear(d_i, dc),
                                                  nn.Linear(d_plan, dc), nn.Linear(d_m, dc))
        self.cell_h = GatedLRUCell(d_z + 3 * dc + 1, d_h, lru_c, (0.8, 0.995), ffn_mult)
        self.cell_i = GatedLRUCell(2 * dc + d_f + d_act + 1, d_i, lru_c, (0.6, 0.98), ffn_mult)
        self.cell_a = GatedLRUCell(4 * dc + d_f, d_a, lru_c, (0.9, 0.9995), ffn_mult)
        self.slow = nn.GRUCell(d_h + d_a + d_i, d_m)

    def init_state(self, batch: int, device=None) -> dict:
        p = next(self.parameters())
        z = lambda d: p.new_zeros(batch, d)  # noqa: E731
        d = self.dims
        return {"s_h": z(d["h"]), "s_a": z(d["a"]), "s_i": z(d["i"]),
                "y_h": z(d["h"]), "y_a": z(d["a"]), "y_i": z(d["i"]),
                "m": z(d["m"]), "acc": z(d["h"] + d["a"] + d["i"]), "tick": 0}

    def step(self, st: dict, z: torch.Tensor, p_user: torch.Tensor, f: torch.Tensor,
             act: torch.Tensor, plan_sum: torch.Tensor) -> dict:
        """One 20 ms tick.  z (B,d_z), p_user (B,1), f (B,d_f), act (B,d_act), plan_sum (B,d_plan)."""
        u_h = torch.cat([z, self.a2h(st["y_a"]), self.i2h(st["y_i"]), self.m2h(st["m"]), p_user], -1)
        s_h, y_h = self.cell_h(u_h, st["s_h"])
        u_i = torch.cat([self.h2i(y_h), self.a2i(st["y_a"]), f, act, p_user], -1)
        s_i, y_i = self.cell_i(u_i, st["s_i"])
        u_a = torch.cat([self.h2a(y_h), self.i2a(y_i), self.p2a(plan_sum), self.m2a(st["m"]), f], -1)
        s_a, y_a = self.cell_a(u_a, st["s_a"])
        acc = st["acc"] + torch.cat([y_h, y_a, y_i], -1)
        tick = st["tick"] + 1
        m = st["m"]
        if tick % self.slow_every == 0:
            m = self.slow(acc / self.slow_every, m)
            acc = torch.zeros_like(acc)
        return {"s_h": s_h, "s_a": s_a, "s_i": s_i, "y_h": y_h, "y_a": y_a, "y_i": y_i,
                "m": m, "acc": acc, "tick": tick}
