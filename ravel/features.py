"""Causal log-mel front end.

Tick t (20 ms) covers samples [hop*t, hop*(t+1)). Its feature is computed from the window of the
last n_fft samples ending at hop*(t+1), i.e. it never uses audio that has not arrived yet.
The batch path (`forward`) and the streaming path (`step`) are numerically identical.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_filterbank(n_fft: int, n_mels: int, sr: int, f_min: float, f_max: float) -> torch.Tensor:
    """HTK-style triangular filterbank, shape (n_fft//2+1, n_mels)."""
    n_freq = n_fft // 2 + 1
    freqs = torch.linspace(0, sr / 2, n_freq)
    m_pts = torch.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    fb = torch.zeros(n_freq, n_mels)
    for m in range(n_mels):
        lo, ce, hi = f_pts[m], f_pts[m + 1], f_pts[m + 2]
        up = (freqs - lo) / (ce - lo)
        down = (hi - freqs) / (hi - ce)
        fb[:, m] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


class CausalLogMel(nn.Module):
    def __init__(self, sample_rate=24000, hop=480, n_fft=1024, n_mels=80, f_min=40.0, f_max=12000.0):
        super().__init__()
        assert n_fft >= hop
        self.hop, self.n_fft, self.n_mels = hop, n_fft, n_mels
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True), persistent=False)
        self.register_buffer("fb", mel_filterbank(n_fft, n_mels, sample_rate, f_min, f_max), persistent=False)

    @property
    def context(self) -> int:
        return self.n_fft - self.hop

    def _frames_to_logmel(self, frames: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft(frames.float() * self.window, dim=-1)
        power = spec.real.pow(2) + spec.imag.pow(2)
        mel = power @ self.fb
        return torch.log(mel + 1e-6)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, N) with N a multiple of hop  ->  (B, N//hop, n_mels)."""
        B, N = wav.shape
        assert N % self.hop == 0, f"length {N} not a multiple of hop {self.hop}"
        x = torch.nn.functional.pad(wav, (self.context, 0))
        frames = x.unfold(-1, self.n_fft, self.hop)  # (B, T, n_fft)
        return self._frames_to_logmel(frames)

    def init_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.context, device=device)

    def step(self, chunk: torch.Tensor, buf: torch.Tensor):
        """chunk (B, hop), buf (B, n_fft-hop) -> feature (B, n_mels), new buf."""
        frame = torch.cat([buf, chunk], dim=-1)
        feat = self._frames_to_logmel(frame)
        return feat, frame[:, self.hop:]
