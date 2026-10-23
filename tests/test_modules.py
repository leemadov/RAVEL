"""Stage 0: module-level shape, causality and streaming-equivalence tests (synthetic tensors only)."""
import torch

from ravel.acoustic_encoder import AcousticPerceptionField
from ravel.acoustic_planner import ShortHorizonPlanner
from ravel.actions import ActionEmbedding, ActionSpace
from ravel.audio_generator import DirectNeuralAudioGenerator
from ravel.baselines import StreamingARBaseline, frame_to_tick_index
from ravel.conversational_core import DualStateCore, GatedLRUCell
from ravel.features import CausalLogMel
from ravel.feedback import SelfMonitoringFeedback
from ravel.influence_field import ReciprocalInfluenceField
from inference.echo import PBFDAF, erle_db


def test_logmel_batch_equals_streaming_and_is_causal():
    mel = CausalLogMel()
    wav = torch.randn(2, 480 * 12) * 0.1
    full = mel(wav)
    assert full.shape == (2, 12, 80)
    buf = mel.init_state(2)
    steps = []
    for t in range(12):
        f, buf = mel.step(wav[:, t * 480:(t + 1) * 480], buf)
        steps.append(f)
    assert torch.allclose(full, torch.stack(steps, 1), atol=1e-4)
    wav2 = wav.clone()
    wav2[:, 480 * 7:] += 1.0  # perturb from tick 7 on
    full2 = mel(wav2)
    assert torch.allclose(full[:, :7], full2[:, :7])
    assert not torch.allclose(full[:, 7], full2[:, 7])


def test_apf_streaming_equivalence_and_causality():
    torch.manual_seed(0)
    apf = AcousticPerceptionField(80, 64, 3, [1, 2, 4], 0.0).eval()
    mic, ref = torch.randn(2, 15, 80), torch.randn(2, 15, 80)
    z, _, aux = apf(mic, ref)
    assert z.shape == (2, 15, 64) and aux["user_vad_logit"].shape == (2, 15)
    st = apf.init_state(2)
    zs = []
    for t in range(15):
        zt, st, _ = apf(mic[:, t:t + 1], ref[:, t:t + 1], st)
        zs.append(zt)
    assert torch.allclose(z, torch.cat(zs, 1), atol=1e-5)
    mic2 = mic.clone()
    mic2[:, 9:] = 5.0
    z2, _, _ = apf(mic2, ref)
    assert torch.allclose(z[:, :9], z2[:, :9], atol=1e-6)
    assert apf.receptive_field == 1 + 2 * (1 + 2 + 4)


def test_lru_cell_stable_and_timescales():
    torch.manual_seed(0)
    cell = GatedLRUCell(8, 16, c=8.0, a_range=(0.9, 0.999))
    tc = cell.time_constants_ticks()
    assert tc.min() > 9.0 and tc.max() < 1001.0  # 1/(1-a) for a in [0.9, 0.999]
    s = torch.zeros(4, 16)
    for _ in range(500):
        s, y = cell(torch.randn(4, 8), s)
    assert torch.isfinite(s).all() and s.abs().max() < 50  # sqrt(1-a^2) normalisation bounds the state


def test_core_step_shapes_and_slow_clock():
    torch.manual_seed(0)
    core = DualStateCore(d_z=32, d_f=16, d_act=8, d_plan=24, d_h=40, d_a=48, d_i=20, d_m=12, d_couple=10,
                         slow_every=4)
    st = core.init_state(3)
    ms = []
    for t in range(9):
        st = core.step(st, torch.randn(3, 32), torch.rand(3, 1), torch.randn(3, 16), torch.randn(3, 8),
                       torch.randn(3, 24))
        ms.append(st["m"].clone())
    assert st["y_h"].shape == (3, 40) and st["y_a"].shape == (3, 48) and st["y_i"].shape == (3, 20)
    # slow memory only changes on ticks 4 and 8 (1-indexed)
    assert torch.equal(ms[0], ms[2]) and not torch.equal(ms[2], ms[3]) and torch.equal(ms[3], ms[6])
    assert not torch.equal(ms[6], ms[7])


def test_rif_shapes_and_action_conditioning():
    torch.manual_seed(0)
    space = ActionSpace(["silent", "yield", "backchannel", "speak", "speak_slow", "speak_fast", "speak_emph"])
    emb = ActionEmbedding(space, 8)
    rif = ReciprocalInfluenceField(16, 16, 8, 8, 8, 32, 16, 2, horizon=6, n_ensemble=3, n_events=2)
    torch.nn.init.normal_(rif.film.weight, std=0.5)
    out = rif(torch.randn(2, 16), torch.randn(2, 16), torch.randn(2, 8), torch.randn(2, 8), emb.table())
    assert out["vad"].shape == (2, 7, 3, 6)
    assert out["onset"].shape == (2, 7, 3, 7)
    assert out["overlap"].shape == (2, 7, 3) and out["cost"].shape == (2, 7, 3)
    assert out["events"].shape == (2, 7, 3, 2)
    assert not torch.allclose(out["cost"][:, 0], out["cost"][:, 3])  # different actions -> different predictions
    s = ReciprocalInfluenceField.summarize(out)
    assert s["p_vad"].shape == (2, 7, 6) and s["cost_std"].shape == (2, 7)
    assert (s["cost_std"] > 0).all()  # ensemble members disagree at init
    critic = ReciprocalInfluenceField(16, 16, 8, 8, 8, 32, 16, 1, 6, 3, 2, listener_heads=False)
    o2 = critic(torch.randn(2, 16), torch.randn(2, 16), torch.randn(2, 8), torch.randn(2, 8), emb.table())
    assert set(o2) == {"cost"} and o2["cost"].shape == (2, 7, 3)


def test_planner_shift_and_revision():
    torch.manual_seed(0)
    pl = ShortHorizonPlanner(n_slots=8, d_slot=32, n_layers=1, n_heads=4, d_a=16, d_i=8, d_act=8, d_rif=5,
                             probe_out=0)
    P = torch.randn(2, 8, 32)
    Ps = pl.shift(P)
    assert torch.equal(Ps[:, :7], P[:, 1:])
    assert torch.allclose(Ps[:, 7], pl.empty.expand(2, 32))
    Pn, info = pl(P, torch.randn(2, 16), torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 5))
    assert Pn.shape == (2, 8, 32) and info["gate"].shape == (2, 8)
    assert ((info["gate"] > 0) & (info["gate"] < 1)).all()
    # P_new is a gated interpolation of the shifted plan and the proposal
    assert torch.allclose(info["shifted"], Ps)
    # different RIF predictions produce a different plan (RIF -> planner pathway exists)
    torch.nn.init.normal_(pl.blocks[0].mod.weight, std=0.1)
    y_a, y_i, a = torch.randn(2, 16), torch.randn(2, 8), torch.randn(2, 8)
    P1, _ = pl(P, y_a, y_i, a, torch.zeros(2, 5))
    P2, _ = pl(P, y_a, y_i, a, torch.ones(2, 5))
    assert not torch.allclose(P1, P2)


def test_dnag_teacher_forcing_matches_generation():
    torch.manual_seed(0)
    g = DirectNeuralAudioGenerator(d_slot=24, d_model=32, n_q=4, vocab=50, frames_per_slot=3).eval()
    slot, prev = torch.randn(5, 24), torch.randint(0, 50, (5, 4))
    toks = g.generate(slot, prev, greedy=True)
    assert toks.shape == (5, 3, 4) and toks.min() >= 0 and toks.max() < 50
    logits = g(slot, prev, toks)
    assert logits.shape == (5, 3, 4, 50)
    assert torch.equal(logits.argmax(-1), toks)  # AR factorisation is consistent between both paths


def test_feedback_shapes():
    fb = SelfMonitoringFeedback(n_q=4, vocab=50, frames_per_slot=3, d_f=16, n_actions=7, ticks_per_slot=2)
    toks, acts = torch.randint(0, 50, (2, 5, 3, 4)), torch.randint(0, 7, (2, 5))
    f = fb(toks, acts)
    assert f.shape == (2, 10, 16)
    assert torch.allclose(f[:, 3], fb.step(toks[:, 1], acts[:, 1], 1), atol=1e-6)


def test_baseline_shapes_and_alignment():
    idx = frame_to_tick_index(6)
    # frame ends at 320, 640, 960, 1280, 1600, 1920 samples -> latest complete 480-sample tick
    assert idx.tolist() == [-1, 0, 1, 1, 2, 3]
    b = StreamingARBaseline(80, 4, 50, 32, 2, 32, 7)
    h = b(torch.randn(2, 8, 80), torch.randn(2, 8, 80), torch.randint(0, 50, (2, 12, 4)))
    assert h.shape == (2, 12, 32)
    lg, tg = b.token_logits(h, torch.randint(0, 50, (2, 12, 4)))
    assert lg.shape == (22, 1, 4, 50) and tg.shape == (22, 1, 4)
    assert b.slot_states(h).shape == (2, 4, 32)


def test_echo_canceller_converges():
    torch.manual_seed(0)
    n = 480 * 300  # 6 s
    far = torch.randn(1, n) * 0.1
    h = torch.randn(1800) * torch.exp(-torch.arange(1800) / 300.0) * 0.3
    echo = torch.nn.functional.conv1d(far[:, None], h.flip(0)[None, None], padding=1799)[:, 0, :n]
    near = echo + torch.randn(1, n) * 1e-4
    aec = PBFDAF(block=480, partitions=4, mu=0.5)
    res = aec.run(far, near)
    erle = erle_db(near[:, -480 * 50:], res[:, -480 * 50:])
    assert erle > 20.0, erle
