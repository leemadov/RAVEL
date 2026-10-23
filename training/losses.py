"""Training losses for RAVEL V2.1.

Index conventions (see docs/SPEC.md): S slots, D = S-1 decisions; decision d chooses the action for
slot d+1 and the RIF predicts listener behaviour in slots d+1 .. d+N.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ravel.audio_generator import token_ce

DEFAULT_WEIGHTS = {
    "dnag": 1.0, "probe": 0.2, "apc": 0.5, "user_vad": 0.5,
    "rif_vad": 1.0, "rif_onset": 0.5, "rif_overlap": 0.5, "rif_events": 0.5, "rif_cost": 1.0,
    "bc": 0.5, "distill": 0.0, "relevance": 0.2,
    "plan_activity": 0.3, "plan_stability": 0.01,
}


def discounted_returns(c: torch.Tensor, gamma: float) -> torch.Tensor:
    """c (B, S) per-slot cost -> G (B, S) with G_s = sum_{k>=0} gamma^k c_{s+k} (truncated at S)."""
    G = torch.zeros_like(c)
    run = torch.zeros_like(c[:, 0])
    for s in range(c.shape[1] - 1, -1, -1):
        run = c[:, s] + gamma * run
        G[:, s] = run
    return G


def rif_targets(batch: dict, N: int, gamma: float, min_tail: int) -> dict:
    """Build horizon targets for every decision d (B, D, ...) from per-slot labels."""
    uv, aa = batch["user_vad"].float(), batch["ai_active"].float()
    B, S = uv.shape
    D = S - 1
    idx = torch.arange(D)[:, None] + 1 + torch.arange(N)[None]  # (D, N) future slot indices
    inr = idx < S
    idc = idx.clamp(max=S - 1)
    vad = uv[:, idc]  # (B, D, N)
    mask = inr[None].expand(B, D, N).float()
    ov = (uv * aa)[:, idc] * mask
    full = inr.all(-1)  # decisions whose whole horizon is inside the sequence
    rel = torch.arange(N)[None, None].expand(B, D, N)  # relative future slot k
    first = torch.where((vad * mask) > 0.5, rel, torch.full_like(rel, N))
    onset = first.min(-1).values  # (B, D) in [0, N]; N = no user activity in the horizon
    onset_valid = (onset < N) | full[None]
    overlap = (ov.sum(-1) > 0).float()
    hfull = full[None].expand(B, D).float()
    # a binary "happens within the horizon" target is known if the horizon is complete OR it already happened
    out = {"vad": vad, "vad_mask": mask, "onset": onset, "onset_mask": onset_valid.float(),
           "overlap": overlap, "overlap_mask": torch.maximum(hfull, overlap), "horizon_mask": hfull}
    if "events" in batch:
        ev = batch["events"].float()[:, idc]  # (B, D, N, n_ev)
        out["events"] = (ev * mask[..., None]).amax(2)
        out["events_mask"] = torch.maximum(hfull[..., None], out["events"])
    if "cost" in batch:
        G = discounted_returns(batch["cost"].float(), gamma)
        out["cost"] = G[:, 1:]  # cost-to-go from slot d+1
        tail_ok = (torch.arange(D) + 1 + min_tail) <= S
        out["cost_mask"] = tail_ok[None].expand(B, D).float()
    return out


def _masked_mean(x, m):
    m = m.float().expand_as(x)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def _gather_action(t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """t (B, D, C, ...) , a (B, D) -> (B, D, ...)."""
    shape = list(a.shape) + [1] + [1] * (t.dim() - 3)
    idx = a.view(shape).expand(*a.shape, 1, *t.shape[3:])
    return t.gather(2, idx).squeeze(2)


def compute_losses(model, batch: dict, res: dict, weights: dict | None = None, dnag_frac: float = 1.0,
                   min_tail: int = 60) -> tuple[torch.Tensor, dict]:
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    cfg = model.cfg
    logs: dict[str, float] = {}
    total = torch.zeros(())
    tokens, actions = batch["ai_tokens"], batch["actions"]
    B, S = actions.shape
    D = S - 1
    valid = batch.get("valid", torch.ones(B, S))[:, 1:].float()  # (B, D)

    def add(name, val):
        nonlocal total
        if w.get(name, 0.0) > 0 and val is not None and torch.isfinite(val):
            total = total + w[name] * val
        if val is not None:
            logs[name] = float(val.detach())

    # --- E. generator (committed slot -> next slot tokens)
    if w["dnag"] > 0:
        sel = (torch.rand(B, D) < dnag_frac) & (valid > 0)
        if sel.sum() == 0:
            sel[0, 0] = True
        logits, tgt = model.dnag_logits(res, tokens, sel)
        add("dnag", token_ce(logits, tgt))
        with torch.no_grad():
            logs["dnag_acc"] = float((logits.argmax(-1) == tgt).float().mean())
            logs["dnag_ce_q0"] = float(token_ce(logits[:, :, :1], tgt[:, :, :1]))
    # --- D. planner multi-horizon probe and activity
    K = res["plan"].shape[2]
    if w["probe"] > 0 and K > 1 and model.planner.probe is not None:
        sel = (torch.rand(B, D) < min(1.0, 4 * dnag_frac / K)) & (valid > 0)
        if sel.sum() == 0:
            sel[0, 0] = True
        pl = model.probe_logits(res, sel)  # (N, K-1, F, pc, V)
        pc = cfg.planner.probe_codebooks
        fidx = torch.arange(D)[None].expand(B, D)[sel]  # decision index per selected row
        fut = fidx[:, None] + 2 + torch.arange(K - 1)[None]  # slots d+2 .. d+K
        inr = (fut < S).float()
        bidx = torch.arange(B)[:, None].expand(B, D)[sel]
        tg = tokens[bidx[:, None], fut.clamp(max=S - 1)][..., :pc]  # (N, K-1, F, pc)
        add("probe", token_ce(pl, tg, inr[:, :, None, None]))
    if w["plan_activity"] > 0 and "ai_active" in batch:
        fut = torch.arange(D)[:, None] + 1 + torch.arange(K)[None]
        m = (fut < S).float()[None] * valid[..., None]
        tgt = batch["ai_active"].float()[:, fut.clamp(max=S - 1)]
        add("plan_activity", _masked_mean(F.binary_cross_entropy_with_logits(res["activity"], tgt, reduction="none"), m))
    if w["plan_stability"] > 0 and K > 1:
        add("plan_stability", res["revision"][..., 1:].mean())
    # --- A. perception
    T = res["mic_mel"].shape[1]
    off = cfg.apf.apc_offset_ticks
    if w["apc"] > 0 and T > off:
        add("apc", F.l1_loss(res["apf"]["apc_pred"][:, :-off], res["mic_mel"][:, off:].detach()))
    if w["user_vad"] > 0 and "user_vad_tick" in batch:
        add("user_vad", F.binary_cross_entropy_with_logits(res["apf"]["user_vad_logit"], batch["user_vad_tick"].float()))
    # --- C. reciprocal influence field (only the logged action has an observed outcome)
    a_next = actions[:, 1:]
    if res.get("rif") is not None and "user_vad" in batch:
        rcfg = cfg.rif
        tg = rif_targets(batch, rcfg.horizon_slots, rcfg.gamma, min_tail)
        E = rcfg.n_ensemble
        boot = (torch.rand(B, D, E) < rcfg.bootstrap).float() * valid[..., None]  # bootstrap mask per member
        rif = {k: _gather_action(v, a_next) for k, v in res["rif"].items()}  # (B, D, E, ...)
        if "vad" in rif:
            l = F.binary_cross_entropy_with_logits(rif["vad"], tg["vad"][:, :, None].expand_as(rif["vad"]), reduction="none")
            add("rif_vad", _masked_mean(l, (tg["vad_mask"][:, :, None] * boot[..., None])))
            l = F.cross_entropy(rif["onset"].flatten(0, 2), tg["onset"][:, :, None].expand(B, D, E).flatten(),
                                reduction="none").view(B, D, E)
            add("rif_onset", _masked_mean(l, tg["onset_mask"][..., None] * boot))
            l = F.binary_cross_entropy_with_logits(rif["overlap"], tg["overlap"][..., None].expand(B, D, E), reduction="none")
            add("rif_overlap", _masked_mean(l, tg["overlap_mask"][..., None] * boot))
            if "events" in tg:
                l = F.binary_cross_entropy_with_logits(rif["events"], tg["events"][:, :, None].expand_as(rif["events"]),
                                                       reduction="none")
                add("rif_events", _masked_mean(l, tg["events_mask"][:, :, None] * boot[..., None]))
        if "cost" in tg:
            l = F.huber_loss(rif["cost"], tg["cost"][..., None].expand(B, D, E), reduction="none", delta=2.0)
            add("rif_cost", _masked_mean(l, tg["cost_mask"][..., None] * boot))
    # --- policy prior: behaviour cloning and/or distillation of RIF scores
    if w["bc"] > 0:
        l = F.cross_entropy(res["prior"].flatten(0, 1), a_next.flatten(), reduction="none").view(B, D)
        add("bc", _masked_mean(l, valid))
    if w["distill"] > 0 and "score" in res:
        target = torch.softmax(-res["score"].detach() / cfg.policy.temperature, -1)
        l = -(target * F.log_softmax(res["prior"], -1)).sum(-1)
        add("distill", _masked_mean(l, valid))
    if w["relevance"] > 0 and "relevance" in batch:
        l = F.binary_cross_entropy_with_logits(res["relevance"], batch["relevance"][:, 1:].float(), reduction="none")
        add("relevance", _masked_mean(l, valid))
    logs["total"] = float(total.detach())
    return total, logs
