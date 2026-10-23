"""Does the short-horizon planner respond to changing incoming audio? (success criterion 31)

Runs a trained model closed-loop in DuplexSim and compares, around user voice onsets, (i) the magnitude of revisions
to uncommitted plan slots k >= 1 and (ii) the plan's predicted AI activity for future slots, against all other
decisions. An event-locked average is reported for offsets -3..+5 slots.

    python evaluation/planner_analysis.py --ckpt checkpoints/sim/e1_C_s0/final.pt --out results/planner_analysis.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", default="/home/user/ravel_data/libri24k")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="rif")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bank, users = get_bank(args.libri, split="test")
    m = load(args.ckpt)
    _, traces = run_model(m, bank, SimConfig(), episode_specs(users, args.n, seed=99), mode=args.mode, return_traces=True)
    offs = range(-3, 6)
    rev = {o: [] for o in offs}
    act = {o: [] for o in offs}
    rev_all, act_all, n_on = [], [], 0
    rev_bi = {o: [] for o in offs}
    act_bi = {o: [] for o in offs}
    for tr in traces:
        ua = np.array([t["user_active"] for t in tr])
        sp = np.array([t["ai_speech"] for t in tr])
        r = np.array([t["revision"][1:].mean() for t in tr])
        a = np.array([t["plan_activity"][1:].mean() for t in tr])
        rev_all += r.tolist()
        act_all += a.tolist()
        onsets = np.where(ua[1:] & ~ua[:-1])[0] + 1
        for s in onsets:
            n_on += 1
            barge = sp[s - 1] if s >= 1 else False  # user onset while the AI was speaking
            for o in offs:
                if 0 <= s + o < len(r):
                    rev[o].append(r[s + o])
                    act[o].append(a[s + o])
                    if barge:
                        rev_bi[o].append(r[s + o])
                        act_bi[o].append(a[s + o])
    f = lambda d: {str(k): (round(float(np.mean(v)), 5) if v else None) for k, v in d.items()}  # noqa: E731
    out = {"ckpt": args.ckpt, "n_episodes": args.n, "n_user_onsets": n_on,
           "n_onsets_while_ai_speaking": len(rev_bi[0]),
           "mean_revision_all_decisions": round(float(np.mean(rev_all)), 5),
           "mean_plan_activity_all_decisions": round(float(np.mean(act_all)), 5),
           "revision_locked_to_user_onset": f(rev), "plan_activity_locked_to_user_onset": f(act),
           "revision_locked_to_onset_while_ai_speaking": f(rev_bi),
           "plan_activity_locked_to_onset_while_ai_speaking": f(act_bi),
           "note": "offset 0 = decision at the end of the first user-active slot (decision d uses audio up to slot d)"}
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
