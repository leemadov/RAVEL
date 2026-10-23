"""Does the policy adapt speaking RATE to the (hidden) listener type?

In DuplexSim the barge-in hazard is lowest when the AI's rate matches the user type (slow/neutral/fast), and the type
is only observable through the user's own speech (speed-perturbed audio). Reports, per user type, the share of the
AI's speech slots spoken slow / normal / fast and the barge-in rate.

    python evaluation/rate_analysis.py --ckpts checkpoints/sim/e1_C_s0/final.pt ... --out results/rate_analysis.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.ablations import load  # noqa: E402
from evaluation.closed_loop import episode_specs, run_model  # noqa: E402
from training.duplex_sim import SimConfig  # noqa: E402
from training.make_sim_data import get_bank  # noqa: E402

CLEAN = {"policy.w_overlap": 0.0, "policy.alpha_relevance": 0.0, "policy.kappa_compute": 0.0, "policy.beta_uncert": 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", default="/home/user/ravel_data/libri24k")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    bank, users = get_bank(a.libri, split="test")
    specs = episode_specs(users, a.n, seed=777)
    out = {}
    for p in a.ckpts:
        m = load(p, CLEAN)
        ms = run_model(m, bank, SimConfig(), specs, mode="rif")
        res = {}
        for ut in ["slow", "neutral", "fast"]:
            idx = [i for i, s in enumerate(specs) if s[1] == ut]
            sp = {r: float(np.mean([ms[i]["frac_" + r] for i in idx])) for r in ["speak_slow", "speak", "speak_fast"]}
            tot = sum(sp.values()) or 1.0
            res[ut] = {"n_episodes": len(idx), "share_of_speech": {k: round(v / tot, 3) for k, v in sp.items()},
                       "bargein_per_episode": round(float(np.mean([ms[i]["bargein"] for i in idx])), 3),
                       "cost": round(float(np.mean([ms[i]["cost"] for i in idx])), 3)}
        out[p] = res
        print(p, json.dumps(res), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
