"""Prepare LibriSpeech (CC BY 4.0) for RAVEL: resample 16 kHz -> 24 kHz, trim silence, encode with the frozen
EnCodec (n_q=4), save one file per speaker. Transcripts are NOT used (audio-only pipeline).

    python scripts/prepare_librispeech.py --root /path/LibriSpeech/dev-clean --out /path/libri24k
Note: the source is 16 kHz, so the 24 kHz audio has no content above 8 kHz.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravel.codec import FrozenCodec  # noqa: E402

SLOT = 960


def trim(x: np.ndarray, sr: int = 24000, rel_db: float = 40.0, margin_s: float = 0.05) -> np.ndarray:
    hop = sr // 100
    n = len(x) // hop
    e = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    act = np.where(e > e.max() - rel_db)[0]
    if len(act) == 0:
        return x
    m = int(margin_s * sr)
    a, b = max(0, act[0] * hop - m), min(len(x), (act[-1] + 1) * hop + m)
    return x[a:b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_minutes", type=float, default=8.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    codec = FrozenCodec(n_q=4, bandwidth=3.0)
    speakers = sorted(os.listdir(args.root))
    t0 = time.time()
    total_s = 0.0
    for spk in speakers:
        dst = os.path.join(args.out, f"{spk}.pt")
        if os.path.exists(dst):
            continue
        files = sorted(glob.glob(os.path.join(args.root, spk, "*", "*.flac")))
        utts, names, toks, dur = [], [], [], 0.0
        for fpath in files:
            x, sr = sf.read(fpath, dtype="float32")
            assert sr == 16000
            y = trim(resample_poly(x, 3, 2).astype(np.float32))
            pad = (-len(y)) % SLOT
            y = np.pad(y, (0, pad))
            peak = np.abs(y).max()
            if peak > 0.99:
                y = y * (0.99 / peak)
            tok = codec.encode(torch.from_numpy(y)[None])[0].to(torch.int16)  # (frames, Q)
            utts.append((y * 32767).astype(np.int16))
            names.append(os.path.basename(fpath)[:-5])
            toks.append(tok)
            dur += len(y) / 24000
            if dur > args.max_minutes * 60:
                break
        torch.save({"sr": 24000, "speaker": spk, "utts": utts, "names": names, "tokens": toks}, dst)
        total_s += dur
        print(f"{spk}: {len(utts)} utts, {dur/60:.1f} min, elapsed {time.time()-t0:.0f}s", flush=True)
    print(f"done: {total_s/3600:.2f} h in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
