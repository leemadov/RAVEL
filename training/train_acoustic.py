"""Stage 1 — acoustic representation learning on LibriSpeech (audio only).

RAVEL:      loss = DNAG token CE (next 40 ms slot) + 0.2 plan probe CE (slots +2..+8) + APC + echo-robust user VAD
Baseline A: loss = frame-level token CE (+ user VAD is not part of the baseline)
Both predict exactly the same target tokens (frames 3 .. 3S-1, i.e. slots 1..S-1) from the same inputs.

    python training/train_acoustic.py --model ravel --steps 500 --out checkpoints/stage1_ravel
    python training/train_acoustic.py --model baseline_a --steps 500 --out checkpoints/stage1_baseA
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravel.audio_generator import token_ce  # noqa: E402
from ravel.baselines import StreamingARBaseline  # noqa: E402
from ravel.config import load_config  # noqa: E402
from ravel.model import RAVEL  # noqa: E402
from training.datasets import LibriAcoustic  # noqa: E402
from training.duplex_sim import split_speakers  # noqa: E402
from training.losses import compute_losses  # noqa: E402
from training.trainer import Trainer, set_seed  # noqa: E402

STAGE1_WEIGHTS = {"dnag": 1.0, "probe": 0.2, "apc": 0.5, "user_vad": 0.5, "bc": 0.0, "relevance": 0.0,
                  "plan_activity": 0.0, "plan_stability": 0.01}


def build_baseline(cfg, d_model, n_layers, d_depth):
    return StreamingARBaseline(cfg.audio.n_mels, cfg.codec.n_q, cfg.codec.vocab, d_model, n_layers, d_depth,
                               len(cfg.actions), cfg.codec.frames_per_slot)


def baseline_loss(model, batch):
    tok = batch["ai_tokens"].flatten(1, 2)  # (B, S*3, Q)
    h = model(batch["mic_mel"], batch["ref_mel"], tok)
    B, Tf = tok.shape[:2]
    sel = torch.zeros(B, Tf - 1, dtype=torch.bool)
    sel[:, 2:] = True  # predictions of frames 3..Tf-1 (same targets as RAVEL: slots 1..S-1)
    logits, tgt = model.token_logits(h, tok, sel)
    loss = token_ce(logits, tgt)
    return loss, {"dnag": float(loss), "dnag_ce_q0": float(token_ce(logits[:, :, :1], tgt[:, :, :1])),
                  "dnag_acc": float((logits.argmax(-1) == tgt).float().mean())}


def ravel_loss(model, batch):
    res = model.forward_train(batch)
    return compute_losses(model, batch, res, STAGE1_WEIGHTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ravel", "baseline_a"], default="ravel")
    ap.add_argument("--config", default="configs/ravel_tiny.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--libri", default="/home/user/ravel_data/libri24k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--slots", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--amp", default="none")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--base_d", type=int, default=1024)
    ap.add_argument("--base_layers", type=int, default=4)
    ap.add_argument("--base_depth", type=int, default=640)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    cfg = load_config(args.config, args.override)
    train_spk, test_spk = split_speakers(args.libri)
    if args.model == "ravel":
        model = RAVEL(cfg)
        loss_fn = ravel_loss
    else:
        model = build_baseline(cfg, args.base_d, args.base_layers, args.base_depth)
        loss_fn = baseline_loss
    n_params = model.n_trainable()
    print(f"{args.model}: {n_params/1e6:.2f}M trainable params", flush=True)
    speak = cfg.actions.index("speak")
    train = LibriAcoustic(args.libri, train_spk, S=args.slots, seed=args.seed, speak_action=speak)
    test = LibriAcoustic(args.libri, test_spk, S=args.slots, seed=1234, speak_action=speak)
    eval_batches = [test.sample(args.batch, np.random.default_rng(1000 + i)) for i in range(8)]  # fixed held-out set

    def eval_fn(m):
        ce, ce0, acc = [], [], []
        for b in eval_batches:
            if args.model == "ravel":
                res = m.forward_train(b)
                lg, tg = m.dnag_logits(res, b["ai_tokens"])
            else:
                tok = b["ai_tokens"].flatten(1, 2)
                h = m(b["mic_mel"], b["ref_mel"], tok)
                sel = torch.zeros(tok.shape[0], tok.shape[1] - 1, dtype=torch.bool)
                sel[:, 2:] = True
                lg, tg = m.token_logits(h, tok, sel)
            ce.append(float(token_ce(lg, tg)))
            ce0.append(float(token_ce(lg[..., :1, :], tg[..., :1])))
            acc.append(float((lg.argmax(-1) == tg).float().mean()))
        return {"test_ce": round(float(np.mean(ce)), 4), "test_ce_q0": round(float(np.mean(ce0)), 4),
                "test_acc": round(float(np.mean(acc)), 4)}

    tr = Trainer(model, args.out, lr=args.lr, total_steps=args.steps, warmup=min(100, args.steps // 5),
                 device=args.device, amp=args.amp, ckpt_every=100)
    resumed = tr.maybe_resume()
    meta = {"model": args.model, "params": n_params, "args": vars(args), "train_speakers": train_spk,
            "test_speakers": test_spk, "resumed": resumed, "torch": torch.__version__}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    if not resumed:
        tr.log({"step": 0, "eval": eval_fn(model.eval())})
        model.train()
    tr.fit(lambda s: train.sample(args.batch, rng), loss_fn, eval_fn, args.eval_every)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
