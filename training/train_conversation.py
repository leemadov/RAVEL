"""Stages 2/3/5 on recorded two-channel dialogue (packed by training/dialog_data.py).

Trains the full model on the AI-equivalent speaker's channel: DNAG token prediction of what that speaker
said next (speech-native responses, audio-only inference), behaviour cloning of when they spoke, echo-robust
human VAD, and the RIF on the other speaker's subsequent behaviour. Optionally initialises from a Stage-1
checkpoint.

    python training/train_conversation.py --data data/dialog_train.pt --test data/dialog_test.pt \
        --out checkpoints/stage2 --init checkpoints/stage1_ravel/last.pt

NOT RUN in this repository: no suitably licensed two-channel spoken QA corpus was prepared (see RESULTS.md).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravel.config import load_config  # noqa: E402
from ravel.model import RAVEL  # noqa: E402
from training.datasets import SimLogs  # noqa: E402
from training.dialog_data import DIALOG_ACTIONS  # noqa: E402
from training.losses import compute_losses  # noqa: E402
from training.train_influence import eval_logs  # noqa: E402
from training.trainer import Trainer, set_seed  # noqa: E402

WEIGHTS = {"dnag": 1.0, "probe": 0.2, "apc": 0.2, "user_vad": 0.5, "bc": 0.5, "relevance": 0.2,
           "rif_events": 0.2, "plan_activity": 0.3}


def load_partial(model, path):
    """Load matching tensors from a checkpoint (e.g. Stage 1 with a different action set)."""
    sd = torch.load(path, weights_only=False, map_location="cpu")["model"]
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    return len(ok), len(own)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ravel_tiny.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--data", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--slots", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", default="bf16" if torch.cuda.is_available() else "none")
    ap.add_argument("--accum", type=int, default=1)
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config, ["actions=" + str(DIALOG_ACTIONS), "core.d_intent=0"] + args.override)
    model = RAVEL(cfg).to(args.device)
    if args.init:
        print("loaded %d/%d tensors from %s" % (*load_partial(model, args.init), args.init))
    train, test = SimLogs(args.data), SimLogs(args.test)
    N = cfg.rif.horizon_slots
    rng = np.random.default_rng(args.seed)
    ev = {k: v[:32] for k, v in test.sequential(min(2 * args.slots, test.ep_slots - N), N).items()}
    tr = Trainer(model, args.out, lr=args.lr, total_steps=args.steps, device=args.device, amp=args.amp, accum=args.accum)
    tr.maybe_resume()
    tr.fit(lambda s: train.sample(args.batch, args.slots, N, rng),
           lambda m, b: compute_losses(m, b, m.forward_train(b), WEIGHTS),
           lambda m: eval_logs(m, {k: v.to(args.device) for k, v in ev.items()}, N, cfg.rif.gamma), eval_every=500)
    tr.save("final.pt")


if __name__ == "__main__":
    main()
