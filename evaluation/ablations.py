"""E1/E3 — closed-loop ablation of the Reciprocal Influence Field in DuplexSim (held-out speakers).

    python evaluation/ablations.py --runs checkpoints/sim --variants B Bcrit C --seeds 0 1 2 \
        --n_eval 200 --out results/e1_closed_loop.json

Checkpoints are expected at <runs>/<tag>_<variant>_s<seed>/final.pt. Reference policies (logging heuristic,
deterministic heuristic, always-silent, random, privileged oracle) are evaluated on the same episode specs.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.closed_loop import episode_specs, run_model, run_reference, summarize  # noqa: E402
from ravel.config import apply_overrides, from_dict  # noqa: E402
from ravel.model import RAVEL  # noqa: E402
from training.duplex_sim import SIM_ACTIONS, HeuristicPolicy, OraclePolicy, SimConfig  # noqa: E402
from training.make_sim_data import get_bank  # noqa: E402


class _Silent:
    def reset(self):
        pass

    def act(self, last, ep):
        return "silent"


class _Random:
    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        pass

    def act(self, last, ep):
        return SIM_ACTIONS[int(self.rng.integers(len(SIM_ACTIONS)))]


REFERENCES = {
    "logging_heuristic(eps=0.1)": lambda r: HeuristicPolicy(r, eps=0.1),
    "heuristic(eps=0)": lambda r: HeuristicPolicy(r, eps=0.0),
    "always_silent": lambda r: _Silent(),
    "random": lambda r: _Random(r),
    "oracle(privileged)": lambda r: OraclePolicy(),
}


def load(path, overrides=None):
    ck = torch.load(path, weights_only=False)
    cfg = from_dict(ck["config"])
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    m = RAVEL(cfg)
    m.load_state_dict(ck["model"])
    return m.eval()


def mean_sem(vals):
    v = np.array(vals, float)
    return {"mean": round(float(v.mean()), 3), "sem_over_seeds": round(float(v.std(ddof=1) / np.sqrt(len(v))), 3) if len(v) > 1 else None,
            "per_seed": [round(float(x), 3) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", default="/home/user/ravel_data/libri24k")
    ap.add_argument("--runs", default="checkpoints/sim")
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--variants", nargs="+", default=["B", "Bcrit", "C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--betas", nargs="+", type=float, default=[0.5])
    ap.add_argument("--no_refs", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    bank, users = get_bank(args.libri, split="test")
    cfg = SimConfig()
    specs = episode_specs(users, args.n_eval, seed=777)
    res = {"n_eval_episodes": args.n_eval, "test_users": users, "episode_slots": cfg.episode_slots,
           "references": {}, "models": {}}
    if not args.no_refs:
        for name, fac in REFERENCES.items():
            res["references"][name] = summarize(run_reference(fac, bank, cfg, specs))
            print(name, res["references"][name]["cost"], flush=True)
    for v in args.variants:
        modes = [("prior", None)] if v == "B" else [("rif", b) for b in args.betas] + [("prior", None)]
        for mode, beta in modes:
            key = f"{v}:{mode}" + (f"(beta={beta})" if beta is not None else "")
            per_seed = []
            for s in args.seeds:
                p = os.path.join(args.runs, f"{args.tag}_{v}_s{s}", "final.pt")
                if not os.path.exists(p):
                    print("missing", p)
                    continue
                m = load(p, {"policy.beta_uncert": beta, "policy.w_overlap": 0.0, "policy.alpha_relevance": 0.0,
                             "policy.kappa_compute": 0.0} if beta is not None else None)
                per_seed.append(summarize(run_model(m, bank, cfg, specs, mode=mode)))
                print(key, s, per_seed[-1]["cost"], flush=True)
            if per_seed:
                res["models"][key] = {k: mean_sem([d[k] for d in per_seed]) for k in per_seed[0] if k != "n_episodes"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v["cost"] for k, v in res["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
