"""DuplexSim tests with a synthetic asset bank (no LibriSpeech / codec needed)."""
import copy

import numpy as np

from training.duplex_sim import (EVENTS, SIM_ACTIONS, SLOT, USER_TYPES, DuplexEpisode, HeuristicPolicy, OraclePolicy,
                                 SimConfig, make_backchannel, pack_episode, run_episode)


class FakeBank:
    def __init__(self):
        rng = np.random.default_rng(0)
        voiced = lambda n: (0.1 * rng.standard_normal(n)).astype(np.float32)  # noqa: E731
        self.users = {"u": [voiced(24000 * 4) for _ in range(5)]}
        self.responses = [{r: voiced(SLOT * n) for r, n in [("slow", 47), ("normal", 40), ("fast", 34)]} for _ in range(4)]
        self.resp_tokens = [{r: np.zeros((len(v[r]) // SLOT * 3, 4), np.int16) for r in v} for v in self.responses]
        self.bc = make_backchannel()
        self.bc_tokens = np.zeros((21, 4), np.int16)
        self.silence_tokens = np.zeros((3, 4), np.int16)

    def user_phrase(self, spk, kind, speed, rng, max_s=None, min_s=None):
        n = int(rng.uniform(min_s or 0.8, max_s or 2.0) * 24000)
        x = self.users[spk][0][:n]
        return np.pad(x, (0, (-len(x)) % SLOT))


def test_episode_runs_with_all_policies():
    bank, cfg = FakeBank(), SimConfig()
    for pol in [HeuristicPolicy(np.random.default_rng(1)), HeuristicPolicy(np.random.default_rng(1), confounded=True),
                OraclePolicy()]:
        recs = run_episode(bank, cfg, np.random.default_rng(2), "u", "neutral", pol)
        assert len(recs) == cfg.episode_slots
        assert all(r["cost"] >= 0 for r in recs)
        p = pack_episode(recs, SIM_ACTIONS)
        assert p["mic"].shape == (cfg.episode_slots * SLOT,) and p["ai_tokens"].shape == (cfg.episode_slots, 3, 4)
        assert p["user_vad_tick"].shape == (2 * cfg.episode_slots,) and p["events"].shape == (cfg.episode_slots, len(EVENTS))


def test_oracle_beats_silence():
    bank, cfg = FakeBank(), SimConfig()

    class Silent:
        def reset(self):
            pass

        def act(self, last, ep):
            return "silent"

    c_or, c_si = [], []
    for s in range(8):
        c_or.append(sum(r["cost"] for r in run_episode(bank, cfg, np.random.default_rng(s), "u", USER_TYPES[s % 3], OraclePolicy())))
        c_si.append(sum(r["cost"] for r in run_episode(bank, cfg, np.random.default_rng(s), "u", USER_TYPES[s % 3], Silent())))
    assert np.mean(c_or) < np.mean(c_si)


def test_snapshot_restore_is_exact():
    bank, cfg = FakeBank(), SimConfig()
    ep = DuplexEpisode(bank, cfg, np.random.default_rng(3), "u", "fast")
    for _ in range(30):
        ep.step("silent")
    snap = ep.snapshot()
    rng_state = copy.deepcopy(ep.rng)
    a = [ep.step(x)["cost"] for x in ["speak", "speak", "silent", "backchannel"] * 5]
    ep.restore(snap)
    ep.rng = rng_state
    b = [ep.step(x)["cost"] for x in ["speak", "speak", "silent", "backchannel"] * 5]
    assert a == b


def test_intent_only_after_user_heard():
    """The AI must not 'know' a new turn exists before any user audio of that turn was emitted."""
    bank, cfg = FakeBank(), SimConfig()
    for seed in range(5):
        ep = DuplexEpisode(bank, cfg, np.random.default_rng(seed), "u", "neutral")
        heard = False
        for _ in range(60):
            r = ep.step("silent")
            heard = heard or r["user_active"]
            if r["intent"][0] > 0:
                assert heard
                break


def test_bargein_hazard_depends_on_rate_type_match():
    cfg = SimConfig(hazard_by_mismatch=(0.01, 0.05, 0.2))
    bank = FakeBank()
    counts = {}
    for rate_action in ["speak_fast", "speak_slow"]:
        n = 0
        for s in range(150):
            ep = DuplexEpisode(bank, cfg, np.random.default_rng(100 + s), "u", "fast")
            ep.ustate, ep.pending, ep.turn_heard, ep.resp, ep.resp_frac = "listening", True, True, 0, 0.0
            ep.patience = 50
            for _ in range(10):
                r = ep.step(rate_action)
                n += r["events"]["bargein"]
                if r["events"]["bargein"]:
                    break
        counts[rate_action] = n
    assert counts["speak_fast"] < counts["speak_slow"]  # matched rate -> fewer barge-ins
