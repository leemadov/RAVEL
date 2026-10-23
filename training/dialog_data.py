"""Pack real two-channel conversations (separate speaker channels, 24 kHz) into the episode format used by
SimLogs, so Stages 2/3/5 can train on recorded dialogue instead of the simulator.

Manifest: JSON lines {"user": "a.wav", "ai": "b.wav"} — time-aligned mono files; "ai" is the speaker whose
role RAVEL learns (the "AI-equivalent speaker"). Labels are derived from the audio only (energy VAD):
  actions    silent / backchannel (AI burst < 0.5 s while the user is active) / speak
  cost       overlap (both active, excluding backchannels) + 0.15 x mutual silence after a user turn
  events     bargein = user onset while the AI is speaking; turn_abandon / reprompt are not observable -> 0
These observational labels are confounded by whatever the recorded speaker knew; see docs/SPEC.md §4.

    python training/dialog_data.py --manifest dialogs.jsonl --out data/dialog_train.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravel.features import CausalLogMel  # noqa: E402
from training.losses import discounted_returns  # noqa: E402

SLOT, TICK = 960, 480
DIALOG_ACTIONS = ["silent", "backchannel", "speak"]


def vad(x: np.ndarray, hop: int, thr_db: float = -42.0) -> np.ndarray:
    n = len(x) // hop
    e = 10 * np.log10((x[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1) + 1e-12)
    return e > thr_db


def label_episode(user: np.ndarray, ai: np.ndarray):
    uv, av = vad(user, SLOT), vad(ai, SLOT)
    S = len(uv)
    actions = np.where(av, 2, 0)
    # short AI bursts during user speech -> backchannel
    s = 0
    while s < S:
        if av[s]:
            e = s
            while e < S and av[e]:
                e += 1
            if e - s <= 12 and uv[max(0, s - 2):e].mean() > 0.5:
                actions[s:e] = 1
            s = e
        else:
            s += 1
    speech = actions == 2
    cost = (uv & speech).astype(np.float32)
    since_user = np.zeros(S, int)
    for i in range(1, S):
        since_user[i] = 0 if uv[i - 1] else since_user[i - 1] + 1
    gap = (~uv) & (~av) & (since_user > 0) & (since_user < 50)
    cost += 0.15 * gap
    events = np.zeros((S, 3), np.uint8)
    onset = uv & ~np.concatenate([[False], uv[:-1]])
    events[:, 0] = onset & np.concatenate([[False], speech[:-1]])
    cost += 2.0 * events[:, 0]
    relevance = (~uv) & np.concatenate([speech[1:], [False]])
    return {"actions": actions.astype(np.int16), "user_vad": uv.astype(np.uint8), "ai_active": av.astype(np.uint8),
            "cost": cost, "events": events, "relevance": relevance.astype(np.uint8),
            "user_vad_tick": vad(user, TICK).astype(np.uint8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episode_slots", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.96)
    ap.add_argument("--min_tail", type=int, default=60)
    ap.add_argument("--echo_db", type=float, default=-15.0, help="simulated playback leakage of the AI channel into the mic")
    args = ap.parse_args()
    import soundfile as sf
    from ravel.codec import FrozenCodec
    codec, mel = FrozenCodec(), CausalLogMel()
    L = args.episode_slots * SLOT
    eps = []
    for line in open(args.manifest):
        item = json.loads(line)
        u, sr1 = sf.read(item["user"], dtype="float32")
        a, sr2 = sf.read(item["ai"], dtype="float32")
        assert sr1 == sr2 == 24000, "expects 24 kHz mono channels"
        n = min(len(u), len(a)) // L * L
        for st in range(0, n, L):
            uu, aa = u[st:st + L], a[st:st + L]
            lab = label_episode(uu, aa)
            tok = codec.encode(torch.from_numpy(aa)[None])[0].view(args.episode_slots, 3, -1).to(torch.int16)
            echo = np.concatenate([np.zeros(96, np.float32), aa[:-96]]) * 10 ** (args.echo_db / 20)
            lab.update({"mic": uu + echo, "ref": aa, "ai_tokens": tok.numpy()})
            eps.append(lab)
    st = lambda k: np.stack([e[k] for e in eps])  # noqa: E731
    with torch.no_grad():
        mic_mel, ref_mel = mel(torch.from_numpy(st("mic"))).half(), mel(torch.from_numpy(st("ref"))).half()
    cost = torch.from_numpy(st("cost"))
    G = discounted_returns(cost, args.gamma)
    Gm = torch.zeros_like(G)
    Gm[:, : args.episode_slots - args.min_tail] = 1
    S = args.episode_slots
    data = {"mic_mel": mic_mel, "ref_mel": ref_mel, "ai_tokens": torch.from_numpy(st("ai_tokens")),
            "actions": torch.from_numpy(st("actions")), "user_vad": torch.from_numpy(st("user_vad")),
            "ai_active": torch.from_numpy(st("ai_active")), "user_vad_tick": torch.from_numpy(st("user_vad_tick")),
            "cost": cost, "cost_to_go": G, "cost_to_go_mask": Gm, "events": torch.from_numpy(st("events")),
            "relevance": torch.from_numpy(st("relevance")), "intent_in": torch.zeros(len(eps), S, 0),
            "meta": {"source": args.manifest, "actions": DIALOG_ACTIONS, "n": len(eps)}}
    torch.save(data, args.out)
    print(f"saved {len(eps)} episodes of {S} slots to {args.out}")


if __name__ == "__main__":
    main()
