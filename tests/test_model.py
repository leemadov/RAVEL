"""Stage 0: integrated-model tests — forward/backward, gradient coverage, streaming equivalence,
end-to-end causality and checkpoint round trip. Synthetic tensors only: passing these says nothing
about conversational ability."""
import os
import tempfile

import torch

from inference.streaming import StreamingRAVEL
from ravel.config import RavelConfig, apply_overrides
from ravel.model import RAVEL
from training.losses import compute_losses, discounted_returns, rif_targets
from tests.conftest import random_batch


def test_tiny_parameter_budget():
    m = RAVEL(RavelConfig())
    n = m.n_trainable()
    assert 30e6 <= n <= 80e6, n  # spec: 30-80M trainable, target ~50M (frozen codec excluded)


def test_forward_shapes(micro_cfg):
    m = RAVEL(micro_cfg)
    b = random_batch(micro_cfg, B=2, S=8)
    res = m.forward_train(b)
    C, E, N, K = len(micro_cfg.actions), micro_cfg.rif.n_ensemble, micro_cfg.rif.horizon_slots, micro_cfg.planner.n_slots
    assert res["y_a"].shape == (2, 7, micro_cfg.core.d_a)
    assert res["plan"].shape == (2, 7, K, micro_cfg.planner.d_slot)
    assert res["rif"]["vad"].shape == (2, 7, C, E, N)
    assert res["prior"].shape == (2, 7, C)
    lg, tg = m.dnag_logits(res, b["ai_tokens"])
    assert lg.shape == (14, 3, 4, 1024) and tg.shape == (14, 3, 4)


def test_backward_reaches_every_module(micro_cfg):
    m = RAVEL(micro_cfg)
    b = random_batch(micro_cfg, B=2, S=8)
    res = m.forward_train(b)
    loss, logs = compute_losses(m, b, res, {"distill": 0.1}, min_tail=2)
    assert torch.isfinite(loss)
    loss.backward()
    for name in ["apf", "feedback", "core", "rif", "policy", "planner", "dnag", "act_emb"]:
        mod = getattr(m, name)
        g = sum(p.grad.abs().sum() for p in mod.parameters() if p.grad is not None)
        assert g > 0, f"no gradient reached {name}"
    no_grad = [n for n, p in m.named_parameters() if p.grad is None]
    assert not no_grad, no_grad


def test_gumbel_mode_sends_gradient_into_policy(micro_cfg):
    m = RAVEL(micro_cfg)
    b = random_batch(micro_cfg, B=2, S=6)
    res = m.forward_train(b, action_mode="gumbel", decide_mode="prior")
    lg, tg = m.dnag_logits(res, b["ai_tokens"])
    torch.nn.functional.cross_entropy(lg.flatten(0, 2), tg.flatten()).backward()
    g = sum(p.grad.abs().sum() for p in m.policy.prior.parameters() if p.grad is not None)
    assert g > 0


def test_streaming_matches_teacher_forced(micro_cfg):
    torch.manual_seed(1)
    m = RAVEL(micro_cfg).eval()
    b = random_batch(micro_cfg, B=2, S=6, with_targets=False)
    with torch.no_grad():
        res = m.forward_train(b)
    sr = StreamingRAVEL(m, codec=None)
    sr.reset(2, init_tokens=b["ai_tokens"][:, 0], init_action=b["actions"][:, 0])
    outs = []
    for t in range(12):
        s = t // 2
        o = sr.process_tick(b["mic_wav"][:, t * 480:(t + 1) * 480], b["ref_wav"][:, t * 480:(t + 1) * 480],
                            forced_action=b["actions"][:, s + 1] if s + 1 < 6 else b["actions"][:, 0],
                            forced_tokens=b["ai_tokens"][:, s + 1] if s + 1 < 6 else b["ai_tokens"][:, 0])
        if o is not None and s + 1 < 6:
            outs.append(o)
    assert len(outs) == 5
    for d, o in enumerate(outs):
        assert torch.allclose(o["core"]["y_a"], res["y_a"][:, d], atol=1e-4)
        assert torch.allclose(o["decision"]["rif"]["cost"], res["rif"]["cost"][:, d], atol=1e-4)
        assert torch.allclose(o["decision"]["P"], res["plan"][:, d], atol=1e-4)


def test_end_to_end_causality(micro_cfg):
    """Changing microphone audio from slot 4 onward must not change any decision made before it."""
    torch.manual_seed(2)
    m = RAVEL(micro_cfg).eval()
    b = random_batch(micro_cfg, B=1, S=8, with_targets=False)
    b2 = {k: v.clone() for k, v in b.items()}
    b2["mic_wav"][:, 4 * 960:] = torch.randn_like(b2["mic_wav"][:, 4 * 960:])
    with torch.no_grad():
        r1, r2 = m.forward_train(b), m.forward_train(b2)
    # decision d uses audio up to the end of slot d -> decisions 0..3 identical, decision 4 differs
    assert torch.allclose(r1["plan"][:, :4], r2["plan"][:, :4], atol=1e-6)
    assert torch.allclose(r1["rif"]["cost"][:, :4], r2["rif"]["cost"][:, :4], atol=1e-6)
    assert not torch.allclose(r1["plan"][:, 4], r2["plan"][:, 4], atol=1e-6)


def test_streaming_generation_runs_and_mutes_silent_actions(micro_cfg):
    m = RAVEL(micro_cfg).eval()
    sr = StreamingRAVEL(m, codec=None, greedy=True)
    sr.reset(1)
    silent = m.space.index("silent")
    for t in range(8):
        o = sr.process_tick(torch.randn(1, 480) * 0.05, forced_action=torch.tensor([silent]))
        if o is not None:
            assert torch.equal(o["tokens"][0], sr.silence)


def test_checkpoint_roundtrip(micro_cfg):
    torch.manual_seed(3)
    m = RAVEL(micro_cfg).eval()
    b = random_batch(micro_cfg, B=1, S=5, with_targets=False)
    with torch.no_grad():
        r1 = m.forward_train(b)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ckpt.pt")
        torch.save({"model": m.state_dict(), "config": micro_cfg.to_dict()}, p)
        ck = torch.load(p, weights_only=False)
        from ravel.config import from_dict
        m2 = RAVEL(from_dict(ck["config"])).eval()
        m2.load_state_dict(ck["model"])
    with torch.no_grad():
        r2 = m2.forward_train(b)
    assert torch.equal(r1["plan"], r2["plan"])


def test_ablation_configs_build(micro_cfg):
    for ov in [{"rif.enabled": False}, {"rif.listener_heads": False}, {"planner.enabled": False}]:
        cfg = apply_overrides(micro_cfg, ov)
        m = RAVEL(cfg)
        b = random_batch(cfg, B=1, S=5)
        res = m.forward_train(b)
        loss, _ = compute_losses(m, b, res, min_tail=2)
        loss.backward()


def test_targets_and_returns():
    c = torch.tensor([[1.0, 0.0, 2.0]])
    G = discounted_returns(c, 0.5)
    assert torch.allclose(G, torch.tensor([[1.5, 1.0, 2.0]]))
    batch = {"user_vad": torch.tensor([[0, 0, 1, 1, 0, 0]]).float(), "ai_active": torch.tensor([[1, 1, 1, 0, 0, 0]]).float()}
    tg = rif_targets(batch, N=3, gamma=0.9, min_tail=0)
    # decision 0 looks at slots 1,2,3 -> user active at 2,3 -> onset k=1, overlap at slot 2
    assert tg["vad"][0, 0].tolist() == [0, 1, 1]
    assert tg["onset"][0, 0] == 1 and tg["overlap"][0, 0] == 1
    # decision 3 looks at slots 4,5,(6 out of range) -> no onset in observed part, horizon not complete
    assert tg["onset"][0, 3] == 3 and tg["onset_mask"][0, 3] == 0 and tg["horizon_mask"][0, 3] == 0
