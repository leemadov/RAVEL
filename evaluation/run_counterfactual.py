"""CLI for E2 (see evaluation/counterfactual.py)."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.ablations import load  # noqa: E402
from evaluation.closed_loop import episode_specs  # noqa: E402
from evaluation.counterfactual import counterfactual_eval, summarize_cf  # noqa: E402
from training.duplex_sim import HeuristicPolicy, SimConfig  # noqa: E402
from training.make_sim_data import get_bank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", default="/home/user/ravel_data/libri24k")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    bank, users = get_bank(args.libri, split="test")
    cfg = SimConfig()
    specs = episode_specs(users, args.n_episodes, seed=4242)
    out = {}
    for p in args.ckpts:
        m = load(p)
        rows = counterfactual_eval(m, bank, cfg, specs, lambda r: HeuristicPolicy(r, eps=0.1),
                                   decision_every=args.every, R=args.R)
        out[p] = {"beta0": summarize_cf(rows, 0.0), "beta0.5": summarize_cf(rows, 0.5)}
        print(p, json.dumps(out[p], indent=1), flush=True)
        torch.save(rows, os.path.splitext(args.out)[0] + "_" + p.replace("/", "_") + "_rows.pt")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
