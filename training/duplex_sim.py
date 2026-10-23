"""DuplexSim — a controlled two-party turn-taking environment built from REAL speech (LibriSpeech, CC BY 4.0).

Purpose: test the Reciprocal Influence Field hypothesis where the ground truth is known. The *listener*
(simulated user) reacts to the AI's acoustic actions through hand-written stochastic rules, so:
  * action -> listener effects are causal by construction and can be measured by intervention
    (clone the simulator state, apply each action, roll out);
  * logging policies can be made randomised (unconfounded) or confounded (they see hidden state).
This is a toy world. Results here do NOT establish anything about real human listeners.

Time unit: one slot = 40 ms = 960 samples at 24 kHz.

User (listener) model
  turn        1-3 phrases. Non-final phrases are cut from the middle of an utterance (they stop
              mid-sentence) except with prob p_complete_nonfinal (ambiguous complete sentence); the final
              phrase ends at a real utterance end. Pauses between phrases: U(pause_slots).
  type        fast / neutral / slow; the user's own speech is speed-perturbed accordingly (an acoustic cue).
  reactions   - AI starts full speech inside a hesitation pause -> user abandons the rest of the turn with
                p_abandon (event turn_abandon) else resumes as scheduled (overlap if AI keeps talking)
              - AI backchannels >= bc_min slots inside a pause -> remaining pause is halved
              - turn complete and the AI does not respond within patience -> user re-prompts (event)
              - while the AI delivers its response, per-slot barge-in hazard depends on the match between
                the AI's speaking rate and the user's type (event bargein; user speaks over the AI)
Costs per slot (the "interaction appropriateness" signal; lower is better): overlap, backchannel overlap,
gap (user waiting for a response), dead air in hesitation pauses, irrelevant AI speech, and one-off event
costs (turn_abandon, reprompt, bargein).
"""
from __future__ import annotations

import copy
import glob
import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.signal import lfilter, resample_poly

SLOT = 960
SR = 24000
SIM_ACTIONS = ["silent", "backchannel", "speak", "speak_slow", "speak_fast"]
RATE_OF = {"speak": "normal", "speak_slow": "slow", "speak_fast": "fast"}
EVENTS = ["bargein", "turn_abandon", "reprompt"]
USER_TYPES = ["slow", "neutral", "fast"]


@dataclass
class SimConfig:
    episode_slots: int = 200
    pre_slots: tuple = (3, 12)
    n_phrase_probs: tuple = (0.35, 0.4, 0.25)
    p_complete_nonfinal: float = 0.25
    phrase_s: tuple = (0.8, 2.0)
    pause_slots: tuple = (6, 20)
    patience_slots: tuple = (25, 45)
    post_gap_slots: tuple = (4, 12)
    p_abandon: float = 0.4
    bc_min_slots: int = 3
    # barge-in hazard per slot of AI speech, indexed by |rate_index - type_index| (0 = matched)
    hazard_by_mismatch: tuple = (0.003, 0.012, 0.03)
    user_speed: dict = field(default_factory=lambda: {"slow": 0.88, "neutral": 1.0, "fast": 1.12})
    costs: dict = field(default_factory=lambda: {
        "overlap": 1.0, "bc_overlap": 0.1, "gap": 0.15, "dead_air": 0.05, "irrelevant": 0.3,
        "turn_abandon": 4.0, "reprompt": 3.0, "bargein": 2.0})
    echo_gain_db: tuple = (-20.0, -6.0)
    noise_db: tuple = (-62.0, -48.0)
    vad_db: float = -42.0  # slot energy threshold (dBFS) for "voice active"


RATE_IDX = {"slow": 0, "normal": 1, "fast": 2}
TYPE_IDX = {"slow": 0, "neutral": 1, "fast": 2}
AI_STRETCH = {"slow": (20, 17), "normal": (1, 1), "fast": (17, 20)}  # resample_poly(up, down): length x up/down


def _fade(x: np.ndarray, n: int = 240) -> np.ndarray:
    x = x.copy()
    n = min(n, len(x) // 2)
    w = np.linspace(0, 1, n, dtype=np.float32)
    x[:n] *= w
    x[-n:] *= w[::-1]
    return x


def _pad_slots(x: np.ndarray) -> np.ndarray:
    return np.pad(x, (0, (-len(x)) % SLOT))


def _speed(x: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return x
    up, down = {0.88: (25, 22), 1.12: (25, 28)}.get(round(factor, 2), (100, int(round(100 * factor))))
    return resample_poly(x, up, down).astype(np.float32)


def slot_db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def make_backchannel(f0: float = 190.0) -> np.ndarray:
    """Synthetic 'mm-hm' hum (7 slots = 280 ms): two voiced hums with a rising then falling pitch."""
    sr = SR
    parts = []
    for dur, f_a, f_b in [(0.12, f0 * 0.95, f0 * 1.1), (0.04, 0, 0), (0.12, f0 * 1.15, f0 * 0.9)]:
        n = int(dur * sr)
        if f_a == 0:
            parts.append(np.zeros(n, np.float32))
            continue
        f = np.linspace(f_a, f_b, n)
        ph = 2 * np.pi * np.cumsum(f) / sr
        sig = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 8))
        env = np.sin(np.linspace(0, np.pi, n)) ** 0.7
        parts.append((0.12 * sig * env).astype(np.float32))
    x = np.concatenate(parts)
    return _pad_slots(np.pad(x, (0, 7 * SLOT - len(x))) if len(x) < 7 * SLOT else x[: 7 * SLOT])


class AssetBank:
    """Speech material: user phrases (on the fly) and pre-encoded AI responses / backchannel / silence."""

    def __init__(self, libri_dir: str, user_speakers: list[str], ai_speaker: str, codec=None,
                 n_responses: int = 160, seed: int = 0, cache: str | None = None):
        rng = np.random.default_rng(seed)
        self.users = {}
        for spk in user_speakers:
            d = torch.load(os.path.join(libri_dir, f"{spk}.pt"), weights_only=False)
            self.users[spk] = [u.astype(np.float32) / 32767 for u in d["utts"] if len(u) > 3 * SR]
        if cache and os.path.exists(cache):
            c = torch.load(cache, weights_only=False)
            self.responses, self.resp_tokens = c["responses"], c["resp_tokens"]
            self.bc, self.bc_tokens, self.silence_tokens = c["bc"], c["bc_tokens"], c["silence_tokens"]
            return
        assert codec is not None, "codec required to build the AI bank"
        d = torch.load(os.path.join(libri_dir, f"{ai_speaker}.pt"), weights_only=False)
        ai_utts = [u.astype(np.float32) / 32767 for u in d["utts"] if len(u) > 3 * SR]
        self.responses, self.resp_tokens = [], []
        for i in range(n_responses):
            u = ai_utts[rng.integers(len(ai_utts))]
            L = int(rng.uniform(1.0, 2.4) * SR)
            st = int(rng.integers(0, len(u) - L))
            seg = _fade(u[st:st + L])
            vers, toks = {}, {}
            for r, (up, down) in AI_STRETCH.items():
                y = _pad_slots(resample_poly(seg, up, down).astype(np.float32) if up != down else seg)
                vers[r] = y
                toks[r] = codec.encode(torch.from_numpy(y)[None])[0].to(torch.int16).numpy()
            self.responses.append(vers)
            self.resp_tokens.append(toks)
        self.bc = make_backchannel()
        self.bc_tokens = codec.encode(torch.from_numpy(self.bc)[None])[0].to(torch.int16).numpy()
        self.silence_tokens = codec.silence_slot(3).to(torch.int16).numpy()
        if cache:
            torch.save({"responses": self.responses, "resp_tokens": self.resp_tokens, "bc": self.bc,
                        "bc_tokens": self.bc_tokens, "silence_tokens": self.silence_tokens}, cache)

    def user_phrase(self, spk: str, kind: str, speed: float, rng, max_s: float | None = None,
                    min_s: float | None = None) -> np.ndarray:
        utts = self.users[spk]
        u = utts[rng.integers(len(utts))]
        L = int(rng.uniform(min_s or 0.8, max_s or 2.0) * SR)
        if kind == "final":
            seg = u[len(u) - L:]
        else:  # cut from the middle: must end >= 0.8 s before the utterance end
            hi = len(u) - L - int(0.8 * SR)
            st = int(rng.integers(0, max(hi, 1)))
            seg = u[st:st + L]
        return _pad_slots(_fade(_speed(seg, speed)))


class DuplexEpisode:
    """One simulated conversation. Call step(action_name) once per 40 ms slot."""

    def __init__(self, bank: AssetBank, cfg: SimConfig, rng: np.random.Generator, user_spk: str, user_type: str):
        self.b, self.c, self.rng = bank, cfg, rng
        self.spk, self.utype = user_spk, user_type
        self.speed = cfg.user_speed[user_type]
        self.t = 0
        self.ustate = "pre"
        self.counter = int(rng.integers(*cfg.pre_slots))
        self.phrase = None
        self.ph_ptr = 0
        self.turn_phrases: list[np.ndarray] = []
        self.pending = False
        self.turn_heard = False
        self.resp = None
        self.resp_frac = 0.0
        self.reacted = False
        self.bc_run = 0
        self.bc_in_pause = 0
        self.prev_ai_speech = False
        self.patience = 0
        self.last_rate = "normal"
        # acoustic channel
        g = 10 ** (rng.uniform(*cfg.echo_gain_db) / 20)
        n = 2400
        delay = int(rng.integers(48, 192))
        ir = np.zeros(n, np.float32)
        ir[delay] = 1.0
        tail = rng.standard_normal(n - delay).astype(np.float32) * np.exp(-np.arange(n - delay) / rng.uniform(300, 900))
        ir[delay:] += 0.3 * tail
        self.ir = (g * ir / np.abs(ir).sum() * 2).astype(np.float32)
        self.zi = np.zeros(n - 1, np.float32)
        self.noise = 10 ** (rng.uniform(*cfg.noise_db) / 20)

    # -------------------------------------------------------------- internals
    def _start_turn(self):
        c, rng = self.c, self.rng
        n = int(rng.choice([1, 2, 3], p=c.n_phrase_probs))
        kinds = ["final" if rng.random() < c.p_complete_nonfinal else "nonfinal" for _ in range(n - 1)] + ["final"]
        self.turn_phrases = [self.b.user_phrase(self.spk, k, self.speed, rng, *c.phrase_s[::-1]) for k in kinds]
        self.pending = True
        self.turn_heard = False  # the AI only learns it owes a response once it hears the user speak
        self.resp = int(rng.integers(len(self.b.responses)))
        self.resp_frac = 0.0
        self._next_phrase()

    def _next_phrase(self):
        self.phrase = self.turn_phrases.pop(0)
        self.ph_ptr = 0
        self.ustate = "speaking"

    def _short_phrase(self, state: str):
        self.phrase = self.b.user_phrase(self.spk, "final", self.speed, self.rng, 1.2, 0.5)
        self.ph_ptr = 0
        self.ustate = state

    def snapshot(self):
        return copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in ("b",)})

    def restore(self, snap):
        for k, v in copy.deepcopy(snap).items():
            setattr(self, k, v)

    @property
    def turn_complete(self) -> bool:
        return self.ustate in ("waiting", "listening")

    def intent(self) -> tuple[float, float]:
        """AI-private knowledge in the simulator: (has pending content, fraction of it already delivered)."""
        p = self.pending and self.turn_heard
        return float(p), float(self.resp_frac if p else 0.0)

    def hidden(self) -> dict:
        return {"ustate": self.ustate, "turn_complete": self.turn_complete, "utype": self.utype,
                "pending": self.pending, "resp_frac": self.resp_frac}

    # -------------------------------------------------------------- one slot
    def step(self, action: str, render: bool = True) -> dict:
        c, rng = self.c, self.rng
        cost = 0.0
        ev = dict.fromkeys(EVENTS, 0)
        if self.ustate == "pre" and self.counter <= 0:
            self._start_turn()
        # ---- AI emission for this slot
        ai_speech = action in RATE_OF
        ai_bc = action == "backchannel"
        ai_audio = np.zeros(SLOT, np.float32)
        ai_tok = self.b.silence_tokens
        irrelevant = False
        if ai_speech:
            rate = RATE_OF[action]
            self.last_rate = rate
            r = self.resp if self.resp is not None else 0
            vers, toks = self.b.responses[r][rate], self.b.resp_tokens[r][rate]
            n = len(vers) // SLOT
            j = min(int(self.resp_frac * n), n - 1)
            ai_audio, ai_tok = vers[j * SLOT:(j + 1) * SLOT], toks[3 * j:3 * j + 3]
            if self.pending:
                self.resp_frac += 1.0 / n
            else:
                irrelevant = True
        if ai_bc:
            j = self.bc_run
            if j < 7:
                ai_audio, ai_tok = self.b.bc[j * SLOT:(j + 1) * SLOT], self.b.bc_tokens[3 * j:3 * j + 3]
            self.bc_run += 1
        else:
            self.bc_run = 0
        # ---- user emission for this slot
        user_audio = np.zeros(SLOT, np.float32)
        if self.ustate in ("speaking", "bargein", "reprompting"):
            user_audio = self.phrase[self.ph_ptr * SLOT:(self.ph_ptr + 1) * SLOT]
            self.ph_ptr += 1
        u_act = slot_db(user_audio) > c.vad_db
        if u_act and self.ustate == "speaking":
            self.turn_heard = True
        ai_active = ai_speech or (ai_bc and slot_db(ai_audio) > c.vad_db)
        # ---- per-slot costs
        cc = c.costs
        comp = {"overlap": u_act and ai_speech, "bc_overlap": u_act and ai_bc, "irrelevant": irrelevant,
                "gap": self.ustate in ("waiting", "listening") and not ai_speech and not u_act,
                "dead_air": self.ustate == "hesitating" and not ai_active}
        for k, v in comp.items():
            if v:
                cost += cc[k]
        # ---- user reaction / state update (takes effect from the next slot)
        st = self.ustate
        if st == "pre":
            self.counter -= 1
        elif st == "speaking":
            if self.ph_ptr * SLOT >= len(self.phrase):
                if self.turn_phrases:
                    self.ustate, self.counter = "hesitating", int(rng.integers(*c.pause_slots))
                    self.reacted, self.bc_in_pause = False, 0
                else:
                    self.ustate, self.patience = "waiting", int(rng.integers(*c.patience_slots))
        elif st == "hesitating":
            if ai_speech and not self.reacted:
                self.reacted = True
                if rng.random() < c.p_abandon:
                    ev["turn_abandon"] = 1
                    self.turn_phrases = []
                    self.ustate, self.patience = "listening", int(rng.integers(*c.patience_slots))
            if self.ustate == "hesitating":
                if ai_bc:
                    self.bc_in_pause += 1
                    if self.bc_in_pause == c.bc_min_slots:
                        self.counter = int(math.ceil(self.counter / 2))
                self.counter -= 1
                if self.counter <= 0:
                    self._next_phrase()
        elif st in ("waiting", "listening"):
            if ai_speech and self.pending:
                self.ustate = "listening"
                h = c.hazard_by_mismatch[abs(RATE_IDX[RATE_OF[action]] - TYPE_IDX[self.utype])]
                if rng.random() < h:
                    ev["bargein"] = 1
                    self._short_phrase("bargein")
            elif not ai_speech:
                self.patience -= 1
                if self.patience <= 0:
                    ev["reprompt"] = 1
                    self._short_phrase("reprompting")
        elif st in ("bargein", "reprompting"):
            if self.ph_ptr * SLOT >= len(self.phrase):
                self.ustate, self.patience = "waiting", int(rng.integers(*c.patience_slots))
        elif st == "post":
            self.counter -= 1
            if self.counter <= 0:
                self._start_turn()
        if self.pending and self.resp_frac >= 1.0 - 1e-6:
            self.pending = False
        if not self.pending and self.ustate in ("waiting", "listening"):
            # response fully delivered (possibly while overlapping) -> user's next turn after a gap
            self.ustate, self.counter = "post", int(rng.integers(*c.post_gap_slots))
        for k, v in ev.items():
            cost += cc[k] * v
        relevance = self.pending and self.turn_complete
        mic = None
        if render:
            echo, self.zi = lfilter(self.ir, [1.0], ai_audio, zi=self.zi)
            mic = (user_audio + echo.astype(np.float32)
                   + self.noise * rng.standard_normal(SLOT).astype(np.float32))
        self.prev_ai_speech = ai_speech
        self.t += 1
        return {"user_audio": user_audio, "ai_audio": ai_audio, "mic": mic, "ai_tokens": np.asarray(ai_tok),
                "intent": self.intent(),
                "user_active": u_act, "ai_active": ai_active, "ai_speech": ai_speech, "cost": cost, "events": ev,
                "relevance": relevance, "ustate_before": st, "comp": comp}


# ------------------------------------------------------------------ logging / reference policies
class HeuristicPolicy:
    """Observable-information heuristic with exploration. Sees: user voice activity of the last slot,
    its own last action and whether it has a pending response (all available to a real system).
    confounded=True additionally peeks at hidden simulator state (turn completion, user type)."""

    def __init__(self, rng, eps: float = 0.1, confounded: bool = False, wait=(2, 14), p_yield: float = 0.6,
                 bc_prob: float = 0.15, rate_mode: str = "random", max_hold: int = 4):
        self.rng, self.eps, self.conf = rng, eps, confounded
        self.wait_rng, self.p_yield, self.bc_prob, self.rate_mode = wait, p_yield, bc_prob, rate_mode
        self.max_hold = max_hold
        self.reset()

    def reset(self):
        r = self.rng
        self.wait = int(r.integers(*self.wait_rng))
        self.hold, self.hold_action = 0, None
        self.silence_run = 0
        self.cur = "silent"
        self.rate_action = "speak"

    def _choose_rate(self, hidden):
        if self.conf and self.rng.random() < 0.8:
            return {"slow": "speak_slow", "neutral": "speak", "fast": "speak_fast"}[hidden["utype"]]
        if self.rate_mode == "normal":
            return "speak"
        return ["speak_slow", "speak", "speak_fast"][int(self.rng.integers(3))]

    def act(self, last: dict | None, ep: DuplexEpisode) -> str:
        r = self.rng
        u_act = bool(last and last["user_active"])
        self.silence_run = 0 if u_act else self.silence_run + 1
        hidden = ep.hidden()
        if self.hold > 0:
            self.hold -= 1
            self.cur = self.hold_action
            return self.cur
        if r.random() < self.eps:
            self.hold_action = SIM_ACTIONS[int(r.integers(len(SIM_ACTIONS)))]
            self.hold = int(r.integers(0, self.max_hold + 1))
            self.cur = self.hold_action
            return self.cur
        speaking = self.cur in RATE_OF
        if speaking:
            if not ep.intent()[0]:  # the AI knows when its own response is finished
                self.cur = "silent"
            elif u_act and r.random() < self.p_yield:
                self.cur = "silent"
            # else continue with the same rate
        elif self.cur == "backchannel" and 0 < ep.bc_run < 7:
            pass
        else:
            wait = self.wait
            if self.conf:
                wait = int(r.integers(1, 4)) if hidden["turn_complete"] else int(r.integers(12, 26))
            if u_act:
                self.cur = "silent"
            elif ep.intent()[0] and self.silence_run >= wait:
                self.rate_action = self._choose_rate(hidden)
                self.cur = self.rate_action
            elif self.silence_run == 2 and r.random() < self.bc_prob:
                self.cur = "backchannel"
            else:
                self.cur = "silent"
        return self.cur


class OraclePolicy:
    """Privileged reference (NOT a baseline): sees hidden turn completion and user type."""

    def __init__(self):
        self.cur = "silent"

    def reset(self):
        self.cur = "silent"

    def act(self, last, ep: DuplexEpisode) -> str:
        h = ep.hidden()
        u_act = bool(last and last["user_active"])
        rate = {"slow": "speak_slow", "neutral": "speak", "fast": "speak_fast"}[h["utype"]]
        if h["ustate"] == "hesitating":
            self.cur = "backchannel" if (ep.bc_run < 7 and ep.bc_in_pause < 3) else "silent"
        elif h["pending"] and h["turn_complete"] and not u_act:
            self.cur = rate
        else:
            self.cur = "silent"
        return self.cur


# ------------------------------------------------------------------ episode generation (logged data)
def run_episode(bank, cfg, rng, spk, utype, policy, n_slots=None, render=True):
    ep = DuplexEpisode(bank, cfg, rng, spk, utype)
    policy.reset()
    recs, last = [], None
    for _ in range(n_slots or cfg.episode_slots):
        a = policy.act(last, ep)
        last = ep.step(a, render=render)
        last["action"] = a
        recs.append(last)
    return recs


def pack_episode(recs, actions_vocab: list[str], tick: int = 480) -> dict:
    """Arrays for one logged episode (numpy, compact dtypes)."""
    S = len(recs)
    out = {
        "mic": np.concatenate([r["mic"] for r in recs]).astype(np.float32),
        "ref": np.concatenate([r["ai_audio"] for r in recs]).astype(np.float32),
        "ai_tokens": np.stack([r["ai_tokens"] for r in recs]).astype(np.int16),
        "actions": np.array([actions_vocab.index(r["action"]) for r in recs], np.int16),
        "user_vad": np.array([r["user_active"] for r in recs], np.uint8),
        "ai_active": np.array([r["ai_active"] for r in recs], np.uint8),
        "cost": np.array([r["cost"] for r in recs], np.float32),
        "events": np.array([[r["events"][e] for e in EVENTS] for r in recs], np.uint8),
        "relevance": np.array([r["relevance"] for r in recs], np.uint8),
        "intent": np.array([r["intent"] for r in recs], np.float32),
    }
    ua = np.concatenate([r["user_audio"] for r in recs]).reshape(S * 2, tick)
    out["user_vad_tick"] = (10 * np.log10((ua.astype(np.float64) ** 2).mean(1) + 1e-12) > SimConfig.vad_db).astype(np.uint8)
    return out


def split_speakers(libri_dir: str, ai_speaker: str = "3576", n_test: int = 8, seed: int = 0):
    spk = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(libri_dir, "*.pt")))
    spk = [s for s in spk if s != ai_speaker]
    rng = np.random.default_rng(seed)
    rng.shuffle(spk)
    return sorted(spk[n_test:]), sorted(spk[:n_test])
