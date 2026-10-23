"""Generate listenable audio samples from the Stage-1 RAVEL-Tiny checkpoint (400 CPU steps).

For each held-out LibriSpeech speaker: a short real prompt (forced as the AI's own previous output) is followed by
free-running RAVEL generation through the full streaming path (APF -> core -> planner -> DNAG -> frozen EnCodec).
Also writes the codec round-trip of the *real* continuation, as a reference for what the codec itself can sound like,
and one sample generated from silence with no prompt.

    python scripts/generate_samples.py --ckpt checkpoints/stage1_ravel/last.pt --out results/samples
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.streaming import StreamingRAVEL  # noqa: E402
from ravel.codec import FrozenCodec  # noqa: E402
from ravel.config import from_dict  # noqa: E402
from ravel.model import RAVEL  # noqa: E402
from training.datasets import load_speakers  # noqa: E402
from training.duplex_sim import split_speakers  # noqa: E402

SLOT = 960


@torch.no_grad()
def generate(model, codec, prompt_tok, n_gen, temperature, top_k, seed):
    """prompt_tok (P, 3, Q) or None. Returns tokens (P+n_gen, 3, Q)."""
    torch.manual_seed(seed)
    sr = StreamingRAVEL(model, codec=codec, temperature=temperature, top_k=top_k)
    sr.reset(1)
    speak = torch.tensor([model.space.index("speak")])
    P = 0 if prompt_tok is None else prompt_tok.shape[0]
    out = []
    rng = np.random.default_rng(seed)
    for s in range(P + n_gen):
        for j in range(model.tps):
            played = sr.cur_audio[:, j * 480:(j + 1) * 480]
            mic = 0.2 * played + 1e-3 * torch.from_numpy(rng.standard_normal((1, 480)).astype(np.float32))
            forced = prompt_tok[s + 1][None] if (j == model.tps - 1 and s + 1 < P) else None
            o = sr.process_tick(mic, forced_action=speak, forced_tokens=forced)
        out.append(o["tokens"][0])
    first = prompt_tok[0] if P else codec.silence_slot(3)
    return torch.stack([first] + out[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/stage1_ravel/last.pt")
    ap.add_argument("--libri", default="/home/user/ravel_data/libri24k")
    ap.add_argument("--out", default="results/samples")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--prompt_s", type=float, default=1.2)
    ap.add_argument("--gen_s", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    model = RAVEL(from_dict(ck["config"])).eval()
    model.load_state_dict(ck["model"])
    codec = FrozenCodec()
    _, test_spk = split_speakers(a.libri)
    data = load_speakers(a.libri, test_spk[: a.n])
    P, G = int(a.prompt_s / 0.04), int(a.gen_s / 0.04)
    for i, spk in enumerate(test_spk[: a.n]):
        u, t = next((u, t) for u, t in data[spk] if len(u) >= (P + G) * SLOT)
        tok = t[: 3 * (P + G)].long().view(P + G, 3, -1)
        gen = generate(model, codec, tok[:P], G, a.temperature, a.top_k, seed=i)
        sf.write(f"{a.out}/{i}_speaker{spk}_ravel_continuation.wav", codec.decode(gen.flatten(0, 1)[None])[0].numpy(), 24000)
        sf.write(f"{a.out}/{i}_speaker{spk}_real_codec_reference.wav", codec.decode(tok.flatten(0, 1)[None])[0].numpy(), 24000)
        print("wrote sample", i, spk, flush=True)
    gen = generate(model, codec, None, G, a.temperature, a.top_k, seed=99)
    sf.write(f"{a.out}/{a.n}_from_silence_ravel.wav", codec.decode(gen.flatten(0, 1)[None])[0].numpy(), 24000)
    print("done")


if __name__ == "__main__":
    main()
