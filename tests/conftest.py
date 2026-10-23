import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ravel.config import RavelConfig, apply_overrides, micro_overrides  # noqa: E402


@pytest.fixture
def micro_cfg():
    torch.manual_seed(0)
    return apply_overrides(RavelConfig(), micro_overrides())


def random_batch(cfg, B=2, S=8, seed=0, with_targets=True):
    g = torch.Generator().manual_seed(seed)
    tps, ts = cfg.audio.ticks_per_slot, cfg.audio.tick_samples
    T = S * tps
    b = {
        "mic_wav": torch.randn(B, T * ts, generator=g) * 0.1,
        "ref_wav": torch.randn(B, T * ts, generator=g) * 0.1,
        "ai_tokens": torch.randint(0, cfg.codec.vocab, (B, S, cfg.codec.frames_per_slot, cfg.codec.n_q), generator=g),
        "actions": torch.randint(0, len(cfg.actions), (B, S), generator=g),
    }
    if with_targets:
        b.update({
            "user_vad": (torch.rand(B, S, generator=g) > 0.5).float(),
            "ai_active": (torch.rand(B, S, generator=g) > 0.5).float(),
            "cost": torch.rand(B, S, generator=g),
            "events": (torch.rand(B, S, len(cfg.rif.events), generator=g) > 0.9).float(),
            "relevance": (torch.rand(B, S, generator=g) > 0.5).float(),
            "user_vad_tick": (torch.rand(B, T, generator=g) > 0.5).float(),
        })
    return b
