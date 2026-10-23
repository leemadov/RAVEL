"""Generate logged DuplexSim episodes for RIF / duplex training.

    python training/make_sim_data.py --split train --n 2000 --policy random_heuristic --out data/sim_train.pt
    python training/make_sim_data.py --split test  --n 200  --policy random_heuristic --out data/sim_test.pt
    python training/make_sim_data.py --split train --n 2000 --policy confounded --out data/sim_train_conf.pt

Each episode stores mic/ref log-mels (float16), the AI channel's codec tokens, logged actions, per-slot
labels, AI intent (shifted by one slot so it is causal) and the discounted cost-to-go computed on the
whole episode (masked where fewer than `min_tail` slots remain).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravel.features import CausalLogMel  # noqa: E402
from training.duplex_sim import (SIM_ACTIONS, USER_TYPES, AssetBank, HeuristicPolicy, SimConfig,  # noqa: E402
                                 pack_episode, run_episode, split_speakers)
from training.losses import discounted_returns  # noqa: E402


def get_bank(libri, ai_speaker="3576", cache="data/sim_bank.pt", split="train"):
    train_spk, test_spk = split_speakers(libri, ai_speaker)
    users = train_spk if split == "train" else test_spk
    codec = None
    if not os.path.exists(cache):
        from ravel.codec import FrozenCodec
        codec = FrozenCodec()
    return AssetBank(libri, users, ai_speaker, codec=codec, cache=cache), users


def make_policy(name, rng):
    if name == "random_heuristic":
        return HeuristicPolicy(rng, eps=0.1)
    if name == "confounded":
        return HeuristicPolicy(rng, eps=0.1, confounded=True)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libri", default="/home/user/ravel_data/libri24k")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--policy", default="random_heuristic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gamma", type=float, default=0.96)
    ap.add_argument("--min_tail", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bank, users = get_bank(args.libri, split=args.split)
    cfg = SimConfig()
    mel = CausalLogMel()
    seed = args.seed + (0 if args.split == "train" else 10_000)
    rng = np.random.default_rng(seed)
    pol = make_policy(args.policy, rng)
    eps = []
    t0 = time.time()
    for i in range(args.n):
        spk = users[rng.integers(len(users))]
        utype = USER_TYPES[rng.integers(3)]
        recs = run_episode(bank, cfg, rng, spk, utype, pol)
        p = pack_episode(recs, SIM_ACTIONS)
        p["user_type"] = USER_TYPES.index(utype)
        eps.append(p)
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{args.n} episodes, {time.time()-t0:.0f}s", flush=True)
    S = cfg.episode_slots
    st = lambda k: np.stack([e[k] for e in eps])  # noqa: E731
    mic, ref = torch.from_numpy(st("mic")), torch.from_numpy(st("ref"))
    with torch.no_grad():
        mic_mel = torch.cat([mel(mic[i:i + 64]) for i in range(0, len(mic), 64)]).half()
        ref_mel = torch.cat([mel(ref[i:i + 64]) for i in range(0, len(ref), 64)]).half()
    cost = torch.from_numpy(st("cost"))
    G = discounted_returns(cost, args.gamma)
    G_mask = torch.zeros_like(G)
    G_mask[:, : S - args.min_tail] = 1.0
    intent = torch.from_numpy(st("intent"))
    intent_in = torch.cat([torch.zeros_like(intent[:, :1]), intent[:, :-1]], 1)  # known during slot s = end of s-1
    data = {
        "mic_mel": mic_mel, "ref_mel": ref_mel,
        "ai_tokens": torch.from_numpy(st("ai_tokens")), "actions": torch.from_numpy(st("actions")),
        "user_vad": torch.from_numpy(st("user_vad")), "ai_active": torch.from_numpy(st("ai_active")),
        "user_vad_tick": torch.from_numpy(st("user_vad_tick")),
        "cost": cost, "cost_to_go": G, "cost_to_go_mask": G_mask,
        "events": torch.from_numpy(st("events")), "relevance": torch.from_numpy(st("relevance")),
        "intent_in": intent_in, "user_type": torch.tensor([e["user_type"] for e in eps]),
        "meta": {"policy": args.policy, "split": args.split, "n": args.n, "gamma": args.gamma, "seed": seed,
                 "actions": SIM_ACTIONS, "users": users, "min_tail": args.min_tail},
    }
    torch.save(data, args.out)
    print(f"saved {args.out}: {args.n} episodes x {S} slots in {time.time()-t0:.0f}s; "
          f"mean episode cost {float(cost.sum(1).mean()):.2f}; action freq "
          f"{np.bincount(data['actions'].flatten().numpy(), minlength=len(SIM_ACTIONS)) / data['actions'].numel()}")


if __name__ == "__main__":
    main()
