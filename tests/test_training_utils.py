"""Trainer checkpoint/resume and dialog labelling tests."""
import json
import os
import tempfile

import numpy as np
import torch

from training.dialog_data import label_episode
from training.trainer import Trainer


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(4, 1)


class Crash(Exception):
    pass


def _run(out, steps, resume, crash_at=None):
    torch.manual_seed(0)
    m = Tiny()
    tr = Trainer(m, out, lr=1e-2, total_steps=steps, warmup=1, ckpt_every=2, log_every=1)
    if resume:
        assert tr.maybe_resume()
    g = torch.Generator().manual_seed(0)
    xs = [torch.randn(8, 4, generator=g) for _ in range(10)]

    def get_batch(s):
        if s == crash_at:
            raise Crash()
        return {"x": xs[s]}

    try:
        tr.fit(get_batch, lambda mm, b: (mm.l(b["x"]).pow(2).mean(), {}))
    except Crash:
        pass
    return m.l.weight.detach().clone()


def test_resume_matches_uninterrupted_run():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        w_full = _run(a, 6, False)
        _run(b, 6, False, crash_at=4)  # crash before step 5 (checkpoint exists at step 4)
        w_res = _run(b, 6, True)       # resume and finish
        assert torch.allclose(w_full, w_res, atol=1e-6)
        assert os.path.exists(os.path.join(b, "log.jsonl"))
        json.loads(open(os.path.join(b, "log.jsonl")).readline())


def test_dialog_labels():
    S = 40
    user = np.zeros(S * 960, np.float32)
    ai = np.zeros(S * 960, np.float32)
    rng = np.random.default_rng(0)
    user[0:10 * 960] = 0.1 * rng.standard_normal(10 * 960)      # user speaks slots 0-9
    ai[13 * 960:30 * 960] = 0.1 * rng.standard_normal(17 * 960)  # AI answers slots 13-29
    user[25 * 960:32 * 960] = 0.1 * rng.standard_normal(7 * 960)  # user barges in at 25
    lab = label_episode(user, ai)
    assert lab["actions"][15] == 2 and lab["actions"][5] == 0
    assert lab["events"][25, 0] == 1 and lab["events"][:, 0].sum() == 1  # one barge-in
    assert lab["cost"][26] >= 1.0  # overlap
    assert lab["user_vad_tick"].shape == (2 * S,)
