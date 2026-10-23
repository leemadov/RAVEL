"""Stages 3-5 (simulator) — train RAVEL variants on logged DuplexSim conversations.

Variants (identical except for the listener-modelling component):
  B      RAVEL without RIF: actions come from the policy head trained by behaviour cloning on the logs.
  Bcrit  RIF architecture with ONLY the scalar cost-to-go head (model-free action-value critic).
  C      full RIF: listener-behaviour heads (per-slot user VAD, onset timing, overlap, listener events)
         + cost-to-go head, ensemble uncertainty.

    python training/train_influence.py --variant C --train data/sim_train.pt --test data/sim_test.pt \
        --out checkpoints/sim_C_s0 --seed 0 --steps 1500
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravel.config import load_config, micro_overrides  # noqa: E402
from ravel.model import RAVEL  # noqa: E402
from training.datasets import SimLogs  # noqa: E402
from training.duplex_sim import EVENTS, SIM_ACTIONS  # noqa: E402
from training.losses import compute_losses, rif_targets  # noqa: E402
from training.trainer import Trainer, set_seed  # noqa: E402

SIM_WEIGHTS = {"dnag": 0.2, "probe": 0.0, "apc": 0.2, "user_vad": 0.5, "bc": 0.5, "distill": 0.0,
               "relevance": 0.2, "plan_activity": 0.3, "plan_stability": 0.01}
VARIANTS = {
    "B": {"rif.enabled": False},
    "Bcrit": {"rif.enabled": True, "rif.listener_heads": False},
    "C": {"rif.enabled": True, "rif.listener_heads": True},
}


def sim_config(variant: str, size: str = "micro", extra: list | None = None):
    ov = dict(micro_overrides()) if size == "micro" else {}
    ov.update({"actions": SIM_ACTIONS, "core.d_intent": 2, "rif.events": EVENTS})
    ov.update(VARIANTS[variant])
    cfg = load_config("configs/ravel_tiny.yaml", ov)
    if extra:
        from ravel.config import apply_overrides
        cfg = apply_overrides(cfg, extra)
    return cfg


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


@torch.no_grad()
def eval_logs(model, batch, N, gamma) -> dict:
    """Predictive metrics on held-out logged episodes, for the logged (taken) action only."""
    res = model.forward_train(batch)
    a = batch["actions"][:, 1:]
    out = {}
    tg = rif_targets(batch, N, gamma, 0)
    if res.get("rif") is not None:
        g = {k: v.gather(2, a.view(*a.shape, 1, *([1] * (v.dim() - 3))).expand(*a.shape, 1, *v.shape[3:])).squeeze(2)
             for k, v in res["rif"].items()}
        cm = tg["cost_mask"] > 0
        pred = g["cost"].mean(2)
        out["cost_mae"] = float((pred - tg["cost"]).abs()[cm].mean())
        out["cost_corr"] = float(np.corrcoef(pred[cm].numpy(), tg["cost"][cm].numpy())[0, 1])
        if "vad" in g:
            p = torch.sigmoid(g["vad"]).mean(2)
            m = tg["vad_mask"] > 0
            out["vad_auc"] = auc(p[m].numpy(), tg["vad"][m].numpy())
            out["vad_brier"] = float(((p - tg["vad"]) ** 2)[m].mean())
            out["vad_ece"] = ece(p[m].numpy(), tg["vad"][m].numpy())
            pe = torch.sigmoid(g["events"]).mean(2)
            for i, e in enumerate(EVENTS):
                mm = tg["events_mask"][..., i] > 0
                out[f"{e}_auc"] = auc(pe[..., i][mm].numpy(), tg["events"][..., i][mm].numpy())
            po = torch.sigmoid(g["overlap"]).mean(2)
            mm = tg["overlap_mask"] > 0
            out["overlap_auc"] = auc(po[mm].numpy(), tg["overlap"][mm].numpy())
    out["bc_acc"] = float((res["prior"].argmax(-1) == a).float().mean())
    pu = torch.sigmoid(res["apf"]["user_vad_logit"]).flatten().numpy()
    out["user_vad_tick_auc"] = auc(pu, batch["user_vad_tick"].flatten().numpy())
    return {k: round(v, 4) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--size", default="micro", choices=["micro", "tiny"])
    ap.add_argument("--train", default="data/sim_train.pt")
    ap.add_argument("--test", default="data/sim_test.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--slots", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max_train_episodes", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    cfg = sim_config(args.variant, args.size, args.override)
    model = RAVEL(cfg)
    train, test = SimLogs(args.train), SimLogs(args.test)
    n_train = args.max_train_episodes or train.n_ep
    N, gamma = cfg.rif.horizon_slots, cfg.rif.gamma
    eval_batch = test.sequential(min(args.slots * 2, test.ep_slots - N), N)
    eval_batch = {k: v[:64] for k, v in eval_batch.items()}
    rng = np.random.default_rng(args.seed)

    def get_batch(step):
        return train.sample(args.batch, args.slots, N, rng, rng.integers(n_train, size=args.batch))

    def loss_fn(m, b):
        res = m.forward_train(b)
        return compute_losses(m, b, res, SIM_WEIGHTS, dnag_frac=0.1)

    tr = Trainer(model, args.out, lr=args.lr, total_steps=args.steps, warmup=100, ckpt_every=250, log_every=25)
    resumed = tr.maybe_resume()
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"variant": args.variant, "params": model.n_trainable(), "args": vars(args),
                   "n_train_episodes": n_train, "resumed": resumed}, f, indent=2)
    print(f"variant {args.variant}: {model.n_trainable()/1e6:.2f}M params", flush=True)
    tr.fit(get_batch, loss_fn, lambda m: eval_logs(m, eval_batch, N, gamma), eval_every=500)
    tr.save("final.pt")


if __name__ == "__main__":
    main()
