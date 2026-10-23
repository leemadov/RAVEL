"""Stage 0 — architecture validation on the available device.

Runs: parameter accounting, forward+backward of RAVEL-Tiny on synthetic tensors for a few sequence
lengths, peak memory, step time, optimizer step, checkpoint save/reload equality, streaming tick latency.
Writes results/stage0_<device>.json. Synthetic data only: this validates plumbing, not intelligence.

    python scripts/stage0_validate.py --device cuda --amp bf16
    python scripts/stage0_validate.py --device cpu
"""
import argparse
import json
import os
import resource
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.streaming import StreamingRAVEL  # noqa: E402
from ravel.config import load_config  # noqa: E402
from ravel.model import RAVEL  # noqa: E402
from training.losses import compute_losses  # noqa: E402


def synth_batch(cfg, B, S, device):
    T = S * cfg.audio.ticks_per_slot
    return {
        "mic_wav": torch.randn(B, T * cfg.audio.tick_samples, device=device) * 0.1,
        "ref_wav": torch.randn(B, T * cfg.audio.tick_samples, device=device) * 0.1,
        "ai_tokens": torch.randint(0, cfg.codec.vocab, (B, S, cfg.codec.frames_per_slot, cfg.codec.n_q), device=device),
        "actions": torch.randint(0, len(cfg.actions), (B, S), device=device),
        "user_vad": (torch.rand(B, S, device=device) > 0.5).float(),
        "ai_active": (torch.rand(B, S, device=device) > 0.5).float(),
        "cost": torch.rand(B, S, device=device),
        "events": (torch.rand(B, S, 3, device=device) > 0.9).float(),
        "relevance": (torch.rand(B, S, device=device) > 0.5).float(),
        "user_vad_tick": (torch.rand(B, T, device=device) > 0.5).float(),
    }


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # Linux: KiB -> GiB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ravel_tiny.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--shapes", default="2x25,4x50,8x50", help="BxS list (S slots of 40 ms)")
    ap.add_argument("--ckpt", action="store_true", help="activation checkpointing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)
    cfg = load_config(args.config, {"train.checkpoint_activations": args.ckpt})
    model = RAVEL(cfg).to(dev)
    report = {"device": str(dev), "torch": torch.__version__, "amp": args.amp, "params": model.param_breakdown(),
              "checkpoint_activations": args.ckpt, "runs": []}
    if dev.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = torch.amp.GradScaler(enabled=args.amp == "fp16")
    for shp in args.shapes.split(","):
        B, S = map(int, shp.split("x"))
        batch = synth_batch(cfg, B, S, dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        times = []
        for it in range(3):
            t0 = time.perf_counter()
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                res = model.forward_train(batch)
                loss, logs = compute_losses(model, batch, res, min_tail=5)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        run = {"B": B, "S": S, "seconds_audio_per_seq": S * 0.04, "step_time_s": round(min(times[1:]), 3),
               "loss": round(logs["total"], 3), "finite": bool(torch.isfinite(loss))}
        run["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3) if dev.type == "cuda" else None
        run["process_max_rss_gb"] = round(rss_gb(), 3)
        report["runs"].append(run)
        print(run, flush=True)
    # checkpoint round trip
    model.eval()
    b = synth_batch(cfg, 1, 10, dev)
    with torch.no_grad():
        r1 = model.forward_train(b)["plan"]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "config": cfg.to_dict()}, p)
        size_mb = os.path.getsize(p) / 2**20
        m2 = RAVEL(cfg).to(dev).eval()
        m2.load_state_dict(torch.load(p, weights_only=False)["model"])
    with torch.no_grad():
        r2 = m2.forward_train(b)["plan"]
    report["checkpoint"] = {"identical_outputs": bool(torch.equal(r1, r2)), "file_mb": round(size_mb, 1)}
    # streaming tick latency (model only; codec decode excluded and reported separately in latency.py)
    sr = StreamingRAVEL(model, codec=None)
    sr.reset(1)
    for t in range(100):
        sr.process_tick(torch.randn(1, cfg.audio.tick_samples, device=dev) * 0.05)
    tt = torch.tensor(sr.timings[10:]) * 1000
    report["stream_tick_ms"] = {"mean": round(float(tt.mean()), 2), "p95": round(float(tt.quantile(0.95)), 2),
                                "max": round(float(tt.max()), 2), "note": "decision ticks include RIF+planner+DNAG"}
    print(json.dumps(report, indent=2))
    out = args.out or f"results/stage0_{dev.type}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
