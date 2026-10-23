"""Frozen neural audio codec wrapper (EnCodec 24 kHz, causal, 75 Hz).

Role: FROZEN, PRETRAINED, NOT PART OF RAVEL'S TRAINABLE PARAMETERS. It is used only to
(1) tokenise training audio and (2) turn RAVEL's predicted tokens into a waveform.
Slot mapping: 40 ms slot = 960 samples = exactly 3 frames of hop 320 at 24 kHz.

Licensing note: the facebookresearch/encodec repository (code) is MIT licensed; the PyPI 0.1.1
package metadata lists CC BY-NC 4.0. Treat the pretrained weights as research/non-commercial
until confirmed (see docs/DATA_AND_LICENSES.md).
"""
from __future__ import annotations

import torch


class FrozenCodec:
    def __init__(self, n_q: int = 4, bandwidth: float = 3.0, device: str = "cpu"):
        from encodec import EncodecModel  # imported lazily so unit tests do not need the weights

        self.model = EncodecModel.encodec_model_24khz().to(device).eval()
        self.model.set_target_bandwidth(bandwidth)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_q, self.device = n_q, device
        self.sample_rate, self.hop = 24000, 320
        assert self.model.quantizer.get_num_quantizers_for_bandwidth(self.model.frame_rate, bandwidth) == n_q
        self._silence = None

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, N) float in [-1, 1], N multiple of 320 -> tokens (B, N/320, Q) long."""
        frames = self.model.encode(wav[:, None, :].to(self.device))
        codes = torch.cat([c for c, _ in frames], dim=-1)  # (B, Q, T)
        return codes.transpose(1, 2).contiguous().cpu()

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, T, Q) -> wav (B, T*320)."""
        codes = tokens.transpose(1, 2).to(self.device)
        return self.model.decode([(codes, None)])[:, 0].cpu()

    @torch.no_grad()
    def decode_stream(self, history: torch.Tensor, new: torch.Tensor, context_frames: int = 75) -> torch.Tensor:
        """Approximate streaming decode: decode [last `context_frames` of history ; new] and return only the
        samples of `new`. EnCodec's decoder is causal but contains an LSTM, so a finite context is an
        approximation (measured in tests/test_codec.py)."""
        ctx = history[:, -context_frames:] if history is not None and history.shape[1] else new[:, :0]
        wav = self.decode(torch.cat([ctx, new], 1))
        return wav[:, ctx.shape[1] * self.hop:]

    def silence_slot(self, frames_per_slot: int = 3) -> torch.Tensor:
        """Tokens (F, Q) that the codec assigns to digital silence (taken from the steady state)."""
        if self._silence is None:
            tok = self.encode(torch.zeros(1, 320 * 30))
            self._silence = tok[0, -frames_per_slot:].clone()
        return self._silence
