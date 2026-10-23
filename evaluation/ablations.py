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


def evaluate_variant(paths, bank_val, val_specs, bank_test, test_specs, cfg, settings):
    """settings: list of (key, mode, select, overrides). Tune on validation (training speakers), report test."""
    out = {}
    for s_idx, p in enumerate(paths):
        val = {}
        for key, mode, select, ov in settings:
            m = load(p, ov)
            val[key] = summarize(run_model(m, bank_val, cfg, val_specs, mode=mode, select=select, seed=s_idx))["cost"]
        out[p] = {"val_cost": val}
        for key, mode, select, ov in settings:
            m = load(p, ov)
            out[p][key] = summarize(run_model(m, bank_test, cfg, test_specs, mode=mode, select=select, seed=s_idx))
        groups = {}
        for key, *_ in settings:
            groups.setdefault(key.split("|")[0], []).append(key)
        for g, keys in groups.items():
            best = min(keys, key=lambda k: val[k])
            out[p][g + "|tuned"] = dict(out[p][best], chosen=best)
        print(p, {k: (v["cost"] if isinstance(v, dict) and "cost" in v else v) for k, v in out[p].items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", default="/home/user/ravel_data/libri24k")
    ap.add_argument("--runs", default="checkpoints/sim")
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--variants", nargs="+", default=["B", "Bcrit", "C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--thetas", nargs="+", type=float, default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    ap.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--no_refs", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    bank_test, users_test = get_bank(args.libri, split="test")
    bank_val, users_train = get_bank(args.libri, split="train")
    cfg = SimConfig()
    specs = episode_specs(users_test, args.n_eval, seed=777)
    val_specs = episode_specs(users_train, args.n_val, seed=123)
    res = {"n_eval_episodes": args.n_eval, "n_val_episodes": args.n_val, "test_users": users_test,
           "episode_slots": cfg.episode_slots, "references": {}, "models": {}, "per_checkpoint": {}}
    if not args.no_refs:
        for name, fac in REFERENCES.items():
            res["references"][name] = summarize(run_reference(fac, bank_test, cfg, specs))
            print(name, res["references"][name]["cost"], flush=True)
    clean = {"policy.w_overlap": 0.0, "policy.alpha_relevance": 0.0, "policy.kappa_compute": 0.0}
    for v in args.variants:
        paths = [p for p in (os.path.join(args.runs, f"{args.tag}_{v}_s{s}", "final.pt") for s in args.seeds) if os.path.exists(p)]
        if not paths:
            print("no checkpoints for", v)
            continue
        if v == "B":
            settings = [("prior:greedy", "prior", "greedy", None), ("prior:sample", "prior", "sample", None)]
            settings += [(f"prior:thresh|{t}", "prior", f"thresh{t}", None) for t in args.thetas]
        else:
            settings = [(f"rif:beta|{b}", "rif", "greedy", dict(clean, **{"policy.beta_uncert": b})) for b in args.betas]
        per = evaluate_variant(paths, bank_val, val_specs, bank_test, specs, cfg, settings)
        res["per_checkpoint"].update(per)
        keys = [k for k in per[paths[0]] if k != "val_cost"]
        for k in keys:
            ds = [per[p][k] for p in paths]
            res["models"][f"{v}:{k}"] = {m: mean_sem([d[m] for d in ds]) for m in ds[0] if m not in ("n_episodes", "chosen")}
            if "chosen" in ds[0]:
                res["models"][f"{v}:{k}"]["chosen"] = [d["chosen"] for d in ds]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v["cost"] for k, v in res["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
