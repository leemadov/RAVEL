"""Full-duplex demo loop.

  --wav in.wav   offline "virtual microphone": feeds a 24 kHz file tick by tick, writes what RAVEL would have
                 played to --out (and the mic+output mix to --out_mix). Works without audio hardware.
  --live         real microphone/speaker via the optional `sounddevice` package (not installed by default;
                 untested in this repository's CI container, which has no audio device).

An untrained or briefly trained checkpoint produces babble or silence: this script demonstrates the
streaming mechanics (20 ms listen ticks, 40 ms commits, echo cancellation, self-monitoring), not
conversational ability.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.echo import PBFDAF  # noqa: E402
from inference.streaming import StreamingRAVEL  # noqa: E402
from ravel.codec import FrozenCodec  # noqa: E402
from ravel.config import from_dict, load_config  # noqa: E402
from ravel.model import RAVEL  # noqa: E402


def load_model(ckpt, config):
    if ckpt:
        ck = torch.load(ckpt, weights_only=False, map_location="cpu")
        m = RAVEL(from_dict(ck["config"]))
        m.load_state_dict(ck["model"])
    else:
        m = RAVEL(load_config(config))
    return m.eval()


def run_wav(args):
    import soundfile as sf
    x, sr = sf.read(args.wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == 24000, "resample the input to 24 kHz first"
    model = load_model(args.ckpt, args.config)
    codec = FrozenCodec()
    sr_ = StreamingRAVEL(model, codec=codec, decide_mode=args.mode, temperature=args.temperature)
    sr_.reset(1)
    aec = PBFDAF(block=480, partitions=10) if args.aec else None
    tick = model.cfg.audio.tick_samples
    n = len(x) // tick
    out = np.zeros(n * tick, np.float32)
    actions = []
    for t in range(n):
        mic = torch.from_numpy(x[t * tick:(t + 1) * tick])[None]
        phase = t % model.tps
        playing = sr_.cur_audio[:, phase * tick:(phase + 1) * tick]
        out[t * tick:(t + 1) * tick] = playing[0].numpy()
        if aec is not None:
            mic = aec.process(playing, mic)
        o = sr_.process_tick(mic)
        if o is not None:
            actions.append(model.space.names[int(o["action"][0])])
    sf.write(args.out, out, 24000)
    if args.out_mix:
        sf.write(args.out_mix, 0.5 * x[: len(out)] + 0.5 * out, 24000)
    t = np.array(sr_.timings) * 1000
    print(f"ticks {n}, mean tick compute {t.mean():.1f} ms (p95 {np.quantile(t, .95):.1f}); "
          f"actions: " + ", ".join(f"{a}={actions.count(a)}" for a in model.space.names))


def run_live(args):
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("pip install sounddevice  (requires PortAudio and an audio device)")
    model = load_model(args.ckpt, args.config)
    codec = FrozenCodec()
    sr_ = StreamingRAVEL(model, codec=codec, decide_mode=args.mode, temperature=args.temperature)
    sr_.reset(1)
    aec = PBFDAF(block=480, partitions=10)
    tick = model.cfg.audio.tick_samples
    state = {"t": 0}

    def cb(indata, outdata, frames, time_info, status):
        phase = state["t"] % model.tps
        playing = sr_.cur_audio[:, phase * tick:(phase + 1) * tick]
        outdata[:, 0] = playing[0].numpy()
        mic = aec.process(playing, torch.from_numpy(indata[:, 0].copy())[None])
        sr_.process_tick(mic)
        state["t"] += 1

    with sd.Stream(samplerate=24000, blocksize=tick, channels=1, dtype="float32", callback=cb):
        print("running full duplex; Ctrl+C to stop")
        while True:
            sd.sleep(1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default="configs/ravel_tiny.yaml")
    ap.add_argument("--mode", default="rif", choices=["rif", "prior"])
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--wav", default=None)
    ap.add_argument("--out", default="demo_out.wav")
    ap.add_argument("--out_mix", default=None)
    ap.add_argument("--aec", action="store_true")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    run_live(args) if args.live else run_wav(args)


if __name__ == "__main__":
    main()
