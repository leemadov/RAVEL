"""Datasets.

LibriAcoustic  (Stage 1) — crops of prepared LibriSpeech (24 kHz audio + frozen EnCodec tokens).
    The utterance is the AI's own output channel (ai_tokens; ref = its waveform). The microphone hears
    the AI's playback (echo: delayed, attenuated) + noise, and with probability p_user a *different*
    speaker talking at the same time (user_vad_tick labels by energy). So Stage 1 trains speaking
    (token continuation through APF->core->planner->DNAG), listening-while-speaking (echo-robust human VAD)
    and APC future-mel prediction together. Audio only; no transcripts.

SimLogs (Stages 3-5) — logged DuplexSim episodes (see training/duplex_sim.py), cropped to S slots with
    N slots of label look-ahead so horizon targets near the crop end stay valid.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from ravel.features import CausalLogMel

SLOT = 960


def load_speakers(libri_dir: str, speakers: list[str]) -> dict:
    out = {}
    for s in speakers:
        d = torch.load(os.path.join(libri_dir, f"{s}.pt"), weights_only=False)
        out[s] = [(u, t) for u, t in zip(d["utts"], d["tokens"])]
    return out


class LibriAcoustic:
    def __init__(self, libri_dir: str, speakers: list[str], S: int = 50, p_user: float = 0.5, seed: int = 0,
                 speak_action: int = 3, mel: CausalLogMel | None = None):
        self.data = load_speakers(libri_dir, speakers)
        self.items = [(s, i) for s in self.data for i, (u, _) in enumerate(self.data[s]) if len(u) >= S * SLOT]
        self.S, self.p_user, self.speak = S, p_user, speak_action
        self.rng = np.random.default_rng(seed)
        self.mel = mel or CausalLogMel()
        self.spk = list(self.data)

    def _crop(self, rng):
        s, i = self.items[rng.integers(len(self.items))]
        u, t = self.data[s][i]
        n_slots = len(u) // SLOT
        st = int(rng.integers(0, n_slots - self.S + 1))
        wav = u[st * SLOT:(st + self.S) * SLOT].astype(np.float32) / 32767
        tok = t[st * 3:(st + self.S) * 3].long().view(self.S, 3, -1)
        return s, wav, tok

    def sample(self, B: int, rng: np.random.Generator | None = None) -> dict:
        rng = rng or self.rng
        S, T = self.S, self.S * 2
        ai, toks, mic, uvad = [], [], [], []
        for _ in range(B):
            spk, wav, tok = self._crop(rng)
            ai.append(wav)
            toks.append(tok)
            g = 10 ** (rng.uniform(-20, -6) / 20)
            d = int(rng.integers(48, 240))
            echo = np.concatenate([np.zeros(d, np.float32), wav[:-d]]) * g
            m = echo + 10 ** (rng.uniform(-62, -48) / 20) * rng.standard_normal(len(wav)).astype(np.float32)
            v = np.zeros(T, np.float32)
            if rng.random() < self.p_user:
                other = [x for x in self.spk if x != spk]
                o = other[rng.integers(len(other))]
                ou, _ = self.data[o][rng.integers(len(self.data[o]))]
                L = min(len(ou), len(wav))
                a = int(rng.integers(0, len(ou) - L + 1))
                seg = ou[a:a + L].astype(np.float32) / 32767 * rng.uniform(0.5, 1.0)
                off = int(rng.integers(0, len(wav) - L + 1)) if len(wav) > L else 0
                us = np.zeros(len(wav), np.float32)
                us[off:off + L] = seg
                # user speech only in a random sub-window so that VAD has positives and negatives
                w0, w1 = sorted(rng.integers(0, len(wav), 2))
                us[:w0] = 0
                us[w1:] = 0
                m = m + us
                e = 10 * np.log10((us.reshape(T, -1).astype(np.float64) ** 2).mean(1) + 1e-12)
                v = (e > -42).astype(np.float32)
            mic.append(m)
            uvad.append(v)
        mic_t, ref_t = torch.from_numpy(np.stack(mic)), torch.from_numpy(np.stack(ai))
        with torch.no_grad():
            mic_mel, ref_mel = self.mel(mic_t), self.mel(ref_t)
        return {"mic_mel": mic_mel, "ref_mel": ref_mel, "ai_tokens": torch.stack(toks),
                "actions": torch.full((B, S), self.speak, dtype=torch.long),
                "user_vad_tick": torch.from_numpy(np.stack(uvad))}


class SimLogs:
    """Holds packed episodes with precomputed mels: dict of tensors with leading episode dim."""

    def __init__(self, path: str):
        d = torch.load(path, weights_only=False)
        self.d = d
        self.n_ep, self.ep_slots = d["actions"].shape

    def sample(self, B: int, S: int, look: int, rng: np.random.Generator, eps: np.ndarray | None = None) -> dict:
        d = self.d
        idx = rng.integers(self.n_ep, size=B) if eps is None else eps
        L = S + look
        st = rng.integers(0, self.ep_slots - S + 1, size=len(idx))
        return self._gather(idx, st, S, L)

    def _gather(self, idx, st, S, L):
        d = self.d
        Bn = len(idx)
        sl = lambda x, n: torch.stack([x[e, s:s + n] for e, s in zip(idx, st)])  # noqa: E731
        ext = lambda x: torch.stack([torch.cat([x[e, s:s + L], x.new_zeros((max(0, s + L - self.ep_slots),) + x.shape[2:])])  # noqa: E731
                                     for e, s in zip(idx, st)])
        valid_ext = torch.stack([torch.cat([torch.ones(min(L, self.ep_slots - s)), torch.zeros(max(0, s + L - self.ep_slots))])
                                 for s in st])
        b = {
            "mic_mel": torch.stack([d["mic_mel"][e, 2 * s:2 * (s + S)] for e, s in zip(idx, st)]).float(),
            "ref_mel": torch.stack([d["ref_mel"][e, 2 * s:2 * (s + S)] for e, s in zip(idx, st)]).float(),
            "ai_tokens": sl(d["ai_tokens"], S).long(),
            "actions": sl(d["actions"], S).long(),
            "intent_in": sl(d["intent_in"], S).float(),
            "user_vad_tick": torch.stack([d["user_vad_tick"][e, 2 * s:2 * (s + S)] for e, s in zip(idx, st)]).float(),
            "user_vad": ext(d["user_vad"]).float(), "ai_active": ext(d["ai_active"]).float(),
            "events": ext(d["events"]).float(), "relevance": sl(d["relevance"], S).float(),
            "label_valid": valid_ext,
            "cost_to_go": sl(d["cost_to_go"], S).float(), "cost_to_go_mask": sl(d["cost_to_go_mask"], S).float(),
        }
        del Bn
        return b

    def sequential(self, S: int, look: int):
        """Deterministic evaluation crops covering every episode (start 0)."""
        idx = np.arange(self.n_ep)
        st = np.zeros(self.n_ep, dtype=int)
        return self._gather(idx, st, S, S + look)
