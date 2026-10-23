"""Linear acoustic echo cancellation (optional DSP front end for live microphone use).

Partitioned-block frequency-domain adaptive filter (PBFDAF, overlap-save, gradient-constrained NLMS),
block = one 20 ms tick (480 samples), P partitions -> echo tail of P*20 ms.
The far-end reference is RAVEL's own loudspeaker output. No double-talk detector is included:
adaptation simply continues during double talk (slower, documented limitation). The learned
reference-conditioned APF handles the residual echo.
"""
from __future__ import annotations

import torch


class PBFDAF:
    def __init__(self, block: int = 480, partitions: int = 10, mu: float = 0.5, beta: float = 0.9,
                 delta: float = 1e-6, batch: int = 1):
        self.L, self.P, self.mu, self.beta, self.delta = block, partitions, mu, beta, delta
        nb = block + 1
        self.W = torch.zeros(batch, partitions, nb, dtype=torch.complex64)
        self.X = torch.zeros(batch, partitions, nb, dtype=torch.complex64)
        self.prev = torch.zeros(batch, block)
        self.pow = torch.full((batch, nb), 1e-3)

    @torch.no_grad()
    def process(self, far: torch.Tensor, near: torch.Tensor) -> torch.Tensor:
        """far, near (B, L) -> echo-cancelled near-end block (B, L)."""
        L = self.L
        Xk = torch.fft.rfft(torch.cat([self.prev, far], -1))
        self.prev = far.clone()
        self.X = torch.roll(self.X, 1, dims=1)
        self.X[:, 0] = Xk
        y = torch.fft.irfft((self.X * self.W).sum(1), n=2 * L)[:, L:]
        e = near - y
        E = torch.fft.rfft(torch.cat([torch.zeros_like(e), e], -1))
        self.pow = self.beta * self.pow + (1 - self.beta) * Xk.abs().pow(2)
        G = self.X.conj() * (E / (self.pow + self.delta))[:, None]
        g = torch.fft.irfft(G, n=2 * L)[..., :L]  # gradient constraint (linear, not circular, convolution)
        G = torch.fft.rfft(torch.cat([g, torch.zeros_like(g)], -1))
        self.W = self.W + self.mu * G
        return e

    def run(self, far: torch.Tensor, near: torch.Tensor) -> torch.Tensor:
        """Whole-signal convenience: far, near (B, N) with N a multiple of L."""
        out = [self.process(far[:, i:i + self.L], near[:, i:i + self.L]) for i in range(0, far.shape[1], self.L)]
        return torch.cat(out, -1)


def erle_db(near: torch.Tensor, residual: torch.Tensor) -> float:
    """Echo return loss enhancement (dB) on echo-only segments."""
    return float(10 * torch.log10(near.pow(2).mean() / residual.pow(2).mean().clamp(min=1e-12)))
