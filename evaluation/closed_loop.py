"""Closed-loop (interventional) evaluation in DuplexSim: the policy's actions change what the simulated
listener does next, so this measures behaviour, not just prediction accuracy.

Model policies run through the real streaming path (StreamingRAVEL.process_tick, 20 ms ticks, decision every
40 ms). The AI's speech *content* comes from the simulator bank (not from the DNAG): this experiment
isolates timing / rate / yielding decisions.
"""
from __future__ import annotations

import numpy as np
import torch

from inference.streaming import StreamingRAVEL
from training.duplex_sim import EVENTS, SIM_ACTIONS, DuplexEpisode

COMP = ["overlap", "bc_overlap", "gap", "dead_air", "irrelevant"]


def episode_metrics(recs: list[dict]) -> dict:
    m = {"cost": float(sum(r["cost"] for r in recs))}
    for k in COMP:
        m[k + "_slots"] = float(sum(bool(r["comp"][k]) for r in recs))
    for e in EVENTS:
        m[e] = float(sum(r["events"][e] for r in recs))
    acts = [r["action"] for r in recs]
    for a in SIM_ACTIONS:
        m["frac_" + a] = acts.count(a) / len(acts)
    # response latency: slots from user turn completion ("waiting" entered) to first AI speech slot
    lat, waiting_since = [], None
    for i, r in enumerate(recs):
        if r["ustate_before"] != "waiting":
            waiting_since = None
            continue
        if waiting_since is None:
            waiting_since = i
        if r["ai_speech"]:
            lat.append(i - waiting_since)
            waiting_since = None
    m["response_latency_slots"] = float(np.mean(lat)) if lat else float("nan")
    return m


def summarize(ms: list[dict]) -> dict:
    keys = ms[0].keys()
    out = {}
    for k in keys:
        v = np.array([x[k] for x in ms], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = round(float(v.mean()), 4) if len(v) else float("nan")
        if k == "cost":
            out["cost_sem"] = round(float(v.std(ddof=1) / np.sqrt(len(v))), 4)
    out["n_episodes"] = len(ms)
    return out


def episode_specs(users: list[str], n: int, seed: int = 777) -> list[tuple]:
    rng = np.random.default_rng(seed)
    return [(users[rng.integers(len(users))], ["slow", "neutral", "fast"][rng.integers(3)], int(rng.integers(1 << 30)))
            for _ in range(n)]


def run_reference(policy_factory, bank, cfg, specs) -> list[dict]:
    out = []
    for spk, ut, seed in specs:
        rng = np.random.default_rng(seed)
        ep = DuplexEpisode(bank, cfg, rng, spk, ut)
        pol = policy_factory(np.random.default_rng(seed + 1))
        pol.reset()
        recs, last = [], None
        for _ in range(cfg.episode_slots):
            a = pol.act(last, ep)
            last = ep.step(a, render=False)
            last["action"] = a
            recs.append(last)
        out.append(episode_metrics(recs))
    return out


@torch.no_grad()
def run_model(model, bank, cfg, specs, mode: str = "rif", batch: int = 50, return_traces: bool = False):
    """mode: 'rif' (RIF scores) or 'prior' (learned policy head)."""
    names = model.cfg.actions
    assert names == SIM_ACTIONS, "model must use the simulator action set"
    tps, tick = model.tps, model.cfg.audio.tick_samples
    out, traces = [], []
    for b0 in range(0, len(specs), batch):
        sp = specs[b0:b0 + batch]
        B = len(sp)
        eps = [DuplexEpisode(bank, cfg, np.random.default_rng(seed), spk, ut) for spk, ut, seed in sp]
        sr = StreamingRAVEL(model, codec=None, decide_mode=mode, silence_tokens=torch.from_numpy(bank.silence_tokens).long())
        sr.reset(B)
        acts = ["silent"] * B
        intent_prev = torch.zeros(B, 2)
        recs = [[] for _ in range(B)]
        tr = [[] for _ in range(B)]
        for s in range(cfg.episode_slots):
            rs = [ep.step(a) for ep, a in zip(eps, acts)]
            for i, r in enumerate(rs):
                r["action"] = acts[i]
                recs[i].append(r)
            mic = torch.from_numpy(np.stack([r["mic"] for r in rs])).float()
            ref = torch.from_numpy(np.stack([r["ai_audio"] for r in rs])).float()
            sr.cur_tokens = torch.from_numpy(np.stack([r["ai_tokens"] for r in rs])).long()
            sr.cur_action = torch.tensor([names.index(a) for a in acts])
            o = None
            for j in range(tps):
                o = sr.process_tick(mic[:, j * tick:(j + 1) * tick], ref[:, j * tick:(j + 1) * tick],
                                    forced_tokens=sr.cur_tokens, intent=intent_prev if model.core.d_intent else None)
            intent_prev = torch.tensor([r["intent"] for r in rs]).float()
            acts = [names[i] for i in o["action"].tolist()]
            if return_traces and o["rif"] is not None:
                for i in range(B):
                    tr[i].append({k: v[i].numpy() for k, v in o["rif"].items()})
        out += [episode_metrics(r) for r in recs]
        traces += tr
    return (out, traces) if return_traces else out
