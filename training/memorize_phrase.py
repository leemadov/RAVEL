"""Teach RAVEL to say ONE fixed sentence by memorising a single real recording (audio only, no text input).

The target is LibriSpeech dev-clean 3752-4944-0063 ("IT RELIEVED HIM FOR A WHILE", 2.04 s, CC BY 4.0). Training
episodes are 2.56 s: a random 2–8 slot silent lead, then the sentence on the AI channel, then silence. The microphone
hears the AI's own echo plus noise. At test time RAVEL generates the sentence through the full streaming path
(core -> planner -> DNAG -> frozen EnCodec). Speech onset is forced at slot 5, or left to the policy head.
Scoring uses Whisper small.en as an EXTERNAL judge only; it is not part of RAVEL.

This is memorisation of one clip. It is not sentence generation, and the model does not know what the words mean.

    python training/memorize_phrase.py train --out checkpoints/phrase --steps 600
    python training/memorize_phrase.py eval --ckpt checkpoints/phrase/final.pt --out results/phrase
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravel.config import load_config, micro_overrides  # noqa: E402
from ravel.features import CausalLogMel  # noqa: E402

SLOT, S_EP = 960, 64
UTT, SPK, TEXT = "3752-4944-0063", "3752", "it relieved him for a while"


def load_clip(libri):
    d = torch.load(os.path.join(libri, f"{SPK}.pt"), weights_only=False)
    i = d["names"].index(UTT)
    wav = d["utts"][i].astype(np.float32) / 32767
    tok = d["tokens"][i].long()
    n = tok.shape[0] // 3
    return wav[: n * SLOT], tok[: 3 * n].view(n, 3, -1)


class PhraseEpisodes:
    def __init__(self, libri, silence):
        self.wav, self.tok = load_clip(libri)
        self.n = self.tok.shape[0]
        self.silence = silence
        self.mel = CausalLogMel()

    def batch(self, B, rng):
        mics, refs, toks, acts = [], [], [], []
        for _ in range(B):
            lead = int(rng.integers(2, 9))
            ai = np.zeros(S_EP * SLOT, np.float32)
            ai[lead * SLOT:(lead + self.n) * SLOT] = self.wav
            tok = self.silence[None].repeat(S_EP, 1, 1).clone()
            tok[lead:lead + self.n] = self.tok
            act = np.zeros(S_EP, np.int64)
            act[lead:lead + self.n] = 1
            g = 10 ** (rng.uniform(-20, -8) / 20)
            mic = np.concatenate([np.zeros(96, np.float32), ai[:-96]]) * g \
                + 10 ** (rng.uniform(-60, -48) / 20) * rng.standard_normal(len(ai)).astype(np.float32)
            mics.append(mic); refs.append(ai); toks.append(tok); acts.append(act)  # noqa: E702
        mic, ref = torch.from_numpy(np.stack(mics)), torch.from_numpy(np.stack(refs))
        act = torch.from_numpy(np.stack(acts))
        return {"mic_mel": self.mel(mic), "ref_mel": self.mel(ref), "ai_tokens": torch.stack(toks), "actions": act,
                "ai_active": act.float(), "user_vad_tick": torch.zeros(B, 2 * S_EP)}


def cfg():
    ov = dict(micro_overrides())
    ov.update({"actions": ["silent", "speak"], "rif.enabled": False, "core.d_intent": 0})
    return load_config("configs/ravel_tiny.yaml", ov)


def train(args):
    from ravel.codec import FrozenCodec
    from ravel.model import RAVEL
    from training.losses import compute_losses
    from training.trainer import Trainer, set_seed
    set_seed(args.seed)
    silence = FrozenCodec().silence_slot(3).long()
    ds = PhraseEpisodes(args.libri, silence)
    model = RAVEL(cfg())
    rng = np.random.default_rng(args.seed)
    w = {"dnag": 1.0, "probe": 0.2, "apc": 0.1, "user_vad": 0.0, "bc": 1.0, "relevance": 0.0,
         "plan_activity": 0.3, "plan_stability": 0.01}
    tr = Trainer(model, args.out, lr=1e-3, total_steps=args.steps, warmup=50, ckpt_every=200, log_every=25)
    tr.fit(lambda s: ds.batch(args.batch, rng), lambda m, b: compute_losses(m, b, m.forward_train(b), w))
    tr.save("final.pt")


@torch.no_grad()
def generate(model, codec, n_slots, temperature, seed, forced_start=5):
    from inference.streaming import StreamingRAVEL
    torch.manual_seed(seed)
    sr = StreamingRAVEL(model, codec=codec, temperature=temperature, top_k=50, greedy=temperature <= 0)
    sr.reset(1)
    rng = np.random.default_rng(seed)
    audio = [torch.zeros(SLOT)]
    for s in range(S_EP - 1):
        for j in range(2):
            mic = 0.2 * sr.cur_audio[:, j * 480:(j + 1) * 480] + 1e-3 * torch.from_numpy(
                rng.standard_normal((1, 480)).astype(np.float32))
            forced = None
            if forced_start is not None and j == 1:
                forced = torch.tensor([int(forced_start <= s + 1 < forced_start + n_slots)])
            o = sr.process_tick(mic, forced_action=forced)
        audio.append(o["audio"][0])
    return torch.cat(audio).numpy()


def evaluate(args):
    import soundfile as sf
    import whisper
    from scipy.signal import resample_poly
    from ravel.codec import FrozenCodec
    from ravel.config import from_dict
    from ravel.model import RAVEL
    ck = torch.load(args.ckpt, weights_only=False)
    model = RAVEL(from_dict(ck["config"])).eval()
    model.load_state_dict(ck["model"])
    codec = FrozenCodec()
    wav, tok = load_clip(args.libri)
    judge = whisper.load_model("small.en")
    asr = lambda x: judge.transcribe(resample_poly(x, 2, 3).astype(np.float32), fp16=False, language="en")["text"].strip()  # noqa: E731
    os.makedirs(args.out, exist_ok=True)
    rows = [{"name": "original_recording", "text": asr(wav)},
            {"name": "original_through_codec", "text": asr(codec.decode(tok.flatten(0, 1)[None])[0].numpy())}]
    sf.write(f"{args.out}/original_recording.wav", wav, 24000)
    for name, temp, fs, seed in [("ravel_greedy", 0.0, 5, 0), ("ravel_temp0.5_a", 0.5, 5, 1), ("ravel_temp0.5_b", 0.5, 5, 2),
                                 ("ravel_temp0.8", 0.8, 5, 3), ("ravel_policy_timing_greedy", 0.0, None, 4)]:
        y = generate(model, codec, tok.shape[0], temp, seed, fs)
        sf.write(f"{args.out}/{name}.wav", y, 24000)
        rows.append({"name": name, "temperature": temp, "forced_start": fs, "text": asr(y)})
        print(rows[-1], flush=True)
    for r in rows[:2]:
        print(r)
    with open(f"{args.out}/whisper_transcripts.json", "w") as f:
        json.dump({"target": TEXT, "judge": "openai-whisper small.en (external, evaluation only)", "rows": rows}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--libri", default="/home/user/ravel_data/libri24k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
