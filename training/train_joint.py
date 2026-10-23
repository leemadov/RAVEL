"""Stage 6 — joint fine-tuning with selective freezing / gradual unfreezing.

Starts from a checkpoint, freezes the listed modules for the first `--freeze_steps` steps (protects the
acoustic abilities learned earlier from large early gradients of new heads), then unfreezes with a lower
learning-rate multiplier. Data: packed episodes (simulator logs or dialog_data.py output). Optionally mixes in
Stage-1 LibriSpeech batches (`--libri_mix`) as rehearsal against catastrophic forgetting.

    python training/train_joint.py --init checkpoints/sim_C/final.pt --data data/sim_train.pt \
        --test data/sim_test.pt --out checkpoints/joint --freeze apf dnag --freeze_steps 500
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravel.config import from_dict  # noqa: E402
from ravel.model import RAVEL  # noqa: E402
from training.datasets import SimLogs  # noqa: E402
from training.losses import compute_losses  # noqa: E402
from training.train_influence import SIM_WEIGHTS, eval_logs  # noqa: E402
from training.trainer import Trainer, set_seed  # noqa: E402


def set_frozen(model, names, frozen: bool):
    for n in names:
        for p in getattr(model, n).parameters():
            p.requires_grad_(not frozen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze", nargs="*", default=["apf", "dnag"])
    ap.add_argument("--freeze_steps", type=int, default=500)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--slots", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dnag_weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    ck = torch.load(args.init, weights_only=False)
    model = RAVEL(from_dict(ck["config"]))
    model.load_state_dict(ck["model"])
    train, test = SimLogs(args.data), SimLogs(args.test)
    N = model.cfg.rif.horizon_slots
    rng = np.random.default_rng(args.seed)
    w = dict(SIM_WEIGHTS, dnag=args.dnag_weight)
    ev = {k: v[:32] for k, v in test.sequential(min(2 * args.slots, test.ep_slots - N), N).items()}
    tr = Trainer(model, args.out, lr=args.lr, total_steps=args.steps)  # optimizer sees all params ...
    set_frozen(model, args.freeze, True)  # ... then freeze (frozen params get no grad, so AdamW skips them)

    def loss_fn(m, b):
        if tr.step == args.freeze_steps:
            set_frozen(m, args.freeze, False)  # gradual unfreezing (optimizer already holds all params)
        return compute_losses(m, b, m.forward_train(b), w, dnag_frac=0.25)

    tr.fit(lambda s: train.sample(args.batch, args.slots, N, rng), loss_fn,
           lambda m: eval_logs(m, ev, N, m.cfg.rif.gamma), eval_every=500)
    tr.save("final.pt")


if __name__ == "__main__":
    main()
