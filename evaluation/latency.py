"""Computational latency benchmark for streaming RAVEL.

Protocol (report all numbers together with device, threads and torch version):
  * batch 1, fp32 unless --amp, model in eval mode, torch.inference_mode, 50 warm-up ticks discarded;
  * a "listen tick" = mel + APF + core step (every 20 ms);
  * a "decision tick" = listen tick + RIF over all candidates + policy + planner + DNAG token generation
    (+ streaming codec decode of the 3 new frames with 75 frames of context if --codec);
  * real-time factor (RTF) = (listen + decision time per slot) / 40 ms;  RTF < 1 is required for real time;
  * algorithmic latency is separate: a user event inside slot s can affect output from slot s+1, i.e.
    40-60 ms + compute (+ playback buffering on a real device, not measured here).

    python evaluation/latency.py --device cpu --codec
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.streaming import StreamingRAVEL  # noqa: E402
from ravel.config import load_config, micro_overrides  # noqa: E402
from ravel.model import RAVEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ravel_tiny.yaml")
    ap.add_argument("--micro", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--codec", action="store_true")
    ap.add_argument("--slots", type=int, default=150)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config, micro_overrides() if args.micro else None)
    model = RAVEL(cfg).to(args.device).eval()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, weights_only=False)["model"])
    codec = None
    if args.codec:
        from ravel.codec import FrozenCodec
        codec = FrozenCodec(device=args.device)
    sr = StreamingRAVEL(model, codec=codec, temperature=0.8, top_k=50)
    sr.reset(1)
    listen, decide = [], []
    speak = torch.tensor([model.space.index("speak")])
    with torch.inference_mode():
        for t in range(2 * args.slots):
            x = torch.randn(1, cfg.audio.tick_samples, device=args.device) * 0.05
            t0 = time.perf_counter()
            # force speaking so the DNAG + codec path is always exercised (worst case)
            o = sr.process_tick(x, forced_action=speak)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if t >= 50:
                (decide if o is not None else listen).append(dt)
    L, Dd = np.array(listen), np.array(decide)
    rep = {
        "device": args.device, "threads": args.threads, "torch": torch.__version__, "model": cfg.name,
        "params_M": round(model.n_trainable() / 1e6, 2), "codec_decode": bool(codec),
        "listen_tick_ms": {"mean": round(L.mean(), 2), "p50": round(np.median(L), 2), "p95": round(np.quantile(L, .95), 2)},
        "decision_tick_ms": {"mean": round(Dd.mean(), 2), "p50": round(np.median(Dd), 2), "p95": round(np.quantile(Dd, .95), 2)},
    }
    per_slot = L.mean() + Dd.mean()
    rep["compute_per_40ms_slot_ms"] = round(per_slot, 2)
    rep["real_time_factor"] = round(per_slot / 40.0, 3)
    rep["algorithmic_reaction_latency_ms"] = "40-60 (slot schedule) + decision compute"
    print(json.dumps(rep, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
