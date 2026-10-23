"""E2 — Are the RIF's action-conditioned predictions correct under INTERVENTION?

While the logging policy drives held-out simulator episodes (the model only listens), we stop at decision
points, snapshot simulator + policy, and for every candidate action a run R Monte-Carlo rollouts:
take a for one slot, then follow the logging policy. This gives ground-truth Q^{pi_b}(h, a) (discounted
cost-to-go, the quantity the RIF's cost head estimates) and listener outcomes P(event within N slots | a).
We compare these with the RIF's predictions for all actions, including actions the logger did not take.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from inference.streaming import StreamingRAVEL
from training.duplex_sim import EVENTS, SIM_ACTIONS, DuplexEpisode


def _rollout_value(ep, pol, last, a, R, H, gamma, N, seed, a_logged=None):
    vals, evs, vads = [], [], []
    for r in range(R):
        e2, p2 = copy.deepcopy(ep), copy.deepcopy(pol)
        e2.b = ep.b
        e2.rng = np.random.default_rng(seed + 7919 * r)
        p2.rng = np.random.default_rng(seed + 7919 * r + 1)
        p2.cur = pol.cur
        g, disc, ev, vad = 0.0, 1.0, np.zeros(len(EVENTS)), np.zeros(N)
        rec = e2.step(a, render=False)
        rec["action"] = a
        p2.cur = a  # the policy continues from the intervened action
        if a != a_logged:
            p2.hold = 0  # an intervention cancels any exploration hold the logger was in
        for k in range(H):
            g += disc * rec["cost"]
            disc *= gamma
            if k < N:
                ev = np.maximum(ev, [rec["events"][x] for x in EVENTS])
                vad[k] = rec["user_active"]
            act = p2.act(rec, e2)
            rec = e2.step(act, render=False)
        vals.append(g)
        evs.append(ev)
        vads.append(vad)
    return float(np.mean(vals)), np.mean(evs, 0), np.mean(vads, 0), float(np.std(vals) / np.sqrt(R))


@torch.no_grad()
def counterfactual_eval(model, bank, cfg, specs, policy_factory, decision_every=10, R=16, H=75, gamma=0.96,
                        min_slot=20):
    names = model.cfg.actions
    N = model.cfg.rif.horizon_slots
    tps, tick = model.tps, model.cfg.audio.tick_samples
    rows = []
    for spk, ut, seed in specs:
        ep = DuplexEpisode(bank, cfg, np.random.default_rng(seed), spk, ut)
        pol = policy_factory(np.random.default_rng(seed + 1))
        pol.reset()
        sr = StreamingRAVEL(model, codec=None, decide_mode="rif", silence_tokens=torch.from_numpy(bank.silence_tokens).long())
        sr.reset(1)
        last, intent_prev = None, torch.zeros(1, 2)
        a = pol.act(None, ep)
        for s in range(cfg.episode_slots - H - 2):
            rec = ep.step(a)
            rec["action"] = a
            sr.cur_tokens = torch.from_numpy(rec["ai_tokens"]).long()[None]
            sr.cur_action = torch.tensor([names.index(a)])
            mic, ref = torch.from_numpy(rec["mic"])[None].float(), torch.from_numpy(rec["ai_audio"])[None].float()
            o = None
            for j in range(tps):
                o = sr.process_tick(mic[:, j * tick:(j + 1) * tick], ref[:, j * tick:(j + 1) * tick],
                                    forced_tokens=sr.cur_tokens, forced_action=torch.tensor([0]),
                                    intent=intent_prev if model.core.d_intent else None)
            intent_prev = torch.tensor([rec["intent"]]).float()
            last = rec
            a_next = pol.act(last, ep)
            if s >= min_slot and s % decision_every == 0:
                summ = o["rif"]
                truth = [_rollout_value(ep, pol, last, x, R, H, gamma, N, seed * 1000 + s, a_next) for x in SIM_ACTIONS]
                row = {"spec": (spk, ut, seed), "slot": s, "ustate": ep.ustate, "logged_action": a_next,
                       "true_q": [t[0] for t in truth], "true_q_sem": [t[3] for t in truth],
                       "true_events": [t[1].tolist() for t in truth], "true_vad": [t[2].tolist() for t in truth],
                       "pred_q": summ["cost_mean"][0].tolist(), "pred_q_std": summ["cost_std"][0].tolist()}
                if "p_events" in summ:
                    row["pred_events"] = summ["p_events"][0].tolist()
                    row["pred_vad"] = summ["p_vad"][0].tolist()
                rows.append(row)
            a = a_next
    return rows


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_cf(rows, beta=0.5):
    tq = np.array([r["true_q"] for r in rows])
    pq = np.array([r["pred_q"] for r in rows])
    ps = np.array([r["pred_q_std"] for r in rows])
    logged = np.array([SIM_ACTIONS.index(r["logged_action"]) for r in rows])
    taken_mask = np.zeros_like(tq, dtype=bool)
    taken_mask[np.arange(len(rows)), logged] = True
    choice = (pq + beta * ps).argmin(1)
    best = tq.argmin(1)
    out = {
        "n_decisions": len(rows),
        "mae_all_actions": float(np.abs(pq - tq).mean()),
        "mae_logged_action": float(np.abs(pq - tq)[taken_mask].mean()),
        "mae_untaken_actions": float(np.abs(pq - tq)[~taken_mask].mean()),
        "spearman_across_actions": float(np.nanmean([spearman(p, t) for p, t in zip(pq, tq)])),
        "best_action_agreement": float((choice == best).mean()),
        "regret_vs_true_best": float((tq[np.arange(len(rows)), choice] - tq.min(1)).mean()),
        "regret_of_logged_action": float((tq[np.arange(len(rows)), logged] - tq.min(1)).mean()),
        "mean_true_q_sem": float(np.mean([r["true_q_sem"] for r in rows])),
    }
    if "pred_events" in rows[0]:
        te = np.array([r["true_events"] for r in rows])  # (n, A, n_ev)
        pe = np.array([r["pred_events"] for r in rows])
        out["event_prob_mae"] = {e: float(np.abs(pe[..., i] - te[..., i]).mean()) for i, e in enumerate(EVENTS)}
        tv, pv = np.array([r["true_vad"] for r in rows]), np.array([r["pred_vad"] for r in rows])
        out["user_vad_prob_mae"] = float(np.abs(pv - tv).mean())
    return out
