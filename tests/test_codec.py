"""Frozen codec checks (skipped when the EnCodec package/weights are unavailable)."""
import math

import pytest
import torch

encodec = pytest.importorskip("encodec")


@pytest.fixture(scope="module")
def codec():
    from ravel.codec import FrozenCodec
    try:
        return FrozenCodec(n_q=4, bandwidth=3.0)
    except Exception as e:  # network / weights unavailable
        pytest.skip(f"codec weights unavailable: {e}")


def _tone(n, sr=24000):
    t = torch.arange(n) / sr
    return (0.3 * torch.sin(2 * math.pi * 220 * t) * (1 + 0.5 * torch.sin(2 * math.pi * 3 * t)))[None]


def test_frame_rate_and_slot_mapping(codec):
    tok = codec.encode(_tone(24000))
    assert tok.shape == (1, 75, 4)  # 75 Hz -> 3 frames per 40 ms slot
    assert 960 % codec.hop == 0 and 960 // codec.hop == 3
    assert codec.silence_slot(3).shape == (3, 4)


def test_codec_is_frozen(codec):
    assert all(not p.requires_grad for p in codec.model.parameters())


def test_streaming_decode_close_to_full_decode(codec):
    tok = codec.encode(_tone(24000 * 2))
    full = codec.decode(tok)
    pieces, hist = [], tok[:, :0]
    for s in range(0, tok.shape[1], 3):
        new = tok[:, s:s + 3]
        pieces.append(codec.decode_stream(hist, new, context_frames=75))
        hist = torch.cat([hist, new], 1)
    stream = torch.cat(pieces, -1)
    assert stream.shape == full.shape
    snr = 10 * torch.log10(full.pow(2).mean() / (full - stream).pow(2).mean().clamp(min=1e-12))
    assert snr > 15.0, float(snr)
