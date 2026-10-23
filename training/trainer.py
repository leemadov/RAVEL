"""Minimal, reproducible training loop: AdamW + warmup/cosine, grad clipping, gradient accumulation,
mixed precision, checkpoint/resume (model, optimizer, scheduler, RNG), JSONL logs, periodic evaluation."""
from __future__ import annotations

import json
import math
import os
import random
import resource
import time

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def max_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


class Trainer:
    def __init__(self, model, out_dir: str, lr: float = 3e-4, weight_decay: float = 0.01, warmup: int = 100,
                 total_steps: int = 1000, grad_clip: float = 1.0, accum: int = 1, amp: str = "none",
                 device: str = "cpu", ckpt_every: int = 200, log_every: int = 10):
        self.model, self.out, self.device = model, out_dir, torch.device(device)
        os.makedirs(out_dir, exist_ok=True)
        decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                      {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
        self.total, self.clip, self.accum = total_steps, grad_clip, accum
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
        self.scaler = torch.amp.GradScaler(enabled=amp == "fp16")
        self.ckpt_every, self.log_every = ckpt_every, log_every
        self.step = 0
        self.log_path = os.path.join(out_dir, "log.jsonl")

    def save(self, name: str = "last.pt", extra: dict | None = None):
        cfg = getattr(self.model, "cfg", None)
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                    "step": self.step, "config": cfg.to_dict() if cfg is not None else None,
                    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "py": random.getstate()},
                    **(extra or {})}, os.path.join(self.out, name))

    def maybe_resume(self, name: str = "last.pt") -> bool:
        p = os.path.join(self.out, name)
        if not os.path.exists(p):
            return False
        ck = torch.load(p, weights_only=False, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.step = ck["step"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["py"])
        return True

    def log(self, rec: dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def fit(self, get_batch, loss_fn, eval_fn=None, eval_every: int = 100):
        """get_batch(step) -> batch dict; loss_fn(model, batch) -> (loss, logs); eval_fn(model) -> dict."""
        self.model.train()
        t_last = time.time()
        while self.step < self.total:
            t0 = time.time()
            logs_acc: dict[str, float] = {}
            for _ in range(self.accum):
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in get_batch(self.step).items()}
                with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                    loss, logs = loss_fn(self.model, batch)
                self.scaler.scale(loss / self.accum).backward()
                for k, v in logs.items():
                    logs_acc[k] = logs_acc.get(k, 0.0) + v / self.accum
            self.scaler.unscale_(self.opt)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()
            self.step += 1
            dt = time.time() - t0
            if self.step % self.log_every == 0 or self.step == 1:
                rec = {"step": self.step, "lr": self.sched.get_last_lr()[0], "grad_norm": float(gn), "step_s": round(dt, 3),
                       "rss_gb": round(max_rss_gb(), 2), **{k: round(v, 5) for k, v in logs_acc.items()}}
                if self.device.type == "cuda":
                    rec["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                self.log(rec)
                print(json.dumps(rec), flush=True)
            if eval_fn is not None and (self.step % eval_every == 0 or self.step == self.total):
                self.model.eval()
                with torch.no_grad():
                    ev = eval_fn(self.model)
                self.model.train()
                rec = {"step": self.step, "eval": ev, "elapsed_s": round(time.time() - t_last, 1)}
                self.log(rec)
                print(json.dumps(rec), flush=True)
            if self.step % self.ckpt_every == 0 or self.step == self.total:
                self.save()
        return self
