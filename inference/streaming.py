"""Tick-by-tick streaming inference for RAVEL (full duplex: listen every 20 ms, commit 40 ms of audio per slot).

Timeline for slot s (ticks 2s, 2s+1):
  * the loudspeaker plays the audio committed for slot s (its tokens/action are fed back via SMF)
  * each tick: mic chunk -> causal mel -> APF -> core step
  * after tick 2s+1: RIF scores candidates, policy picks a_{s+1}, planner revises, DNAG emits the 3 codec
    frames of slot s+1, codec decodes 960 samples to be played next.
Non-emitting actions (silent / yield) are hard-gated to the codec's silence tokens, so the selected
action is binding for the next slot.
"""
from __future__ import annotations

import time

import torch

from ravel.influence_field import ReciprocalInfluenceField


class StreamingRAVEL:
    def __init__(self, model, codec=None, decide_mode: str = "rif", temperature: float = 0.8, top_k: int | None = 50,
                 greedy: bool = False, decode_context_frames: int = 75, silence_tokens: torch.Tensor | None = None,
                 select: str = "greedy"):
        self.m = model.eval()
        self.codec = codec
        self.decide_mode, self.temperature, self.top_k, self.greedy = decide_mode, temperature, top_k, greedy
        self.select = select  # action selection: greedy | hier | sample
        self.ctx_frames = decode_context_frames
        c = model.cfg.codec
        if silence_tokens is None:
            silence_tokens = codec.silence_slot(c.frames_per_slot) if codec is not None else \
                torch.zeros(c.frames_per_slot, c.n_q, dtype=torch.long)
        self.silence = silence_tokens
        self.emits = model.space.emits
        self.tick_samples = model.cfg.audio.tick_samples
        self.slot_samples = model.cfg.audio.slot_samples

    @torch.no_grad()
    def reset(self, batch: int = 1, init_tokens: torch.Tensor | None = None, init_action: torch.Tensor | None = None):
        m = self.m
        self.mic_buf = m.mel.init_state(batch)
        self.ref_buf = m.mel.init_state(batch)
        self.apf_state = m.apf.init_state(batch)
        self.core_state = m.core.init_state(batch)
        self.P = m.planner.init_plan(batch)
        self.tick = 0
        self.cur_tokens = init_tokens if init_tokens is not None else self.silence.expand(batch, *self.silence.shape).clone()
        self.cur_action = init_action if init_action is not None else torch.full((batch,), m.space.index("silent"))
        self.cur_audio = torch.zeros(batch, self.slot_samples)
        self.history = self.cur_tokens.clone()  # (B, frames, Q)
        self.timings: list[float] = []

    @torch.no_grad()
    def process_tick(self, mic_chunk: torch.Tensor, ref_chunk: torch.Tensor | None = None,
                     forced_action: torch.Tensor | None = None, forced_tokens: torch.Tensor | None = None,
                     intent: torch.Tensor | None = None):
        """mic_chunk (B, 480). Returns None mid-slot, or a dict for the slot that will be played next."""
        t0 = time.perf_counter()
        m, tps = self.m, self.m.tps
        phase = self.tick % tps
        if ref_chunk is None:
            ref_chunk = self.cur_audio[:, phase * self.tick_samples:(phase + 1) * self.tick_samples]
        mic_f, self.mic_buf = m.mel.step(mic_chunk, self.mic_buf)
        ref_f, self.ref_buf = m.mel.step(ref_chunk, self.ref_buf)
        z, self.apf_state, aux = m.apf(mic_f[:, None], ref_f[:, None], self.apf_state)
        p_user = torch.sigmoid(aux["user_vad_logit"])  # (B, 1)
        f = m.feedback.step(self.cur_tokens, self.cur_action, phase)
        act = m.act_emb(self.cur_action)
        self.core_state = m.core.step(self.core_state, z[:, 0], p_user, f, act, self.P.mean(1), intent)
        self.tick += 1
        if phase != tps - 1:
            self.timings.append(time.perf_counter() - t0)
            return None
        out = m.decide(self.core_state, self.P, self.decide_mode, self.select, forced_action)
        self.P, a = out["P"], out["idx"]
        if forced_tokens is not None:
            tokens = forced_tokens
        else:
            tokens = m.dnag.generate(self.P[:, 0], self.cur_tokens[:, -1], self.temperature, self.top_k, self.greedy)
            mute = ~self.emits[a]
            tokens[mute] = self.silence
        audio = None
        if self.codec is not None:
            audio = self.codec.decode_stream(self.history, tokens, self.ctx_frames)
            self.cur_audio = audio
        self.history = torch.cat([self.history, tokens], 1)[:, -self.ctx_frames:]
        self.cur_tokens, self.cur_action = tokens, a
        summ = ReciprocalInfluenceField.summarize(out["rif"]) if out["rif"] is not None else None
        self.timings.append(time.perf_counter() - t0)
        return {"action": a, "tokens": tokens, "audio": audio, "rif": summ, "p_user": p_user,
                "core": self.core_state, "decision": out}
