#!/usr/bin/env python
"""F5-TTS zero-shot smoke test (workstream 2, engine 2).

F5 conditions on a reference clip + its transcript (prompt_text) and generates
the target sentence. Writes raw output; loudness norm is a separate ffmpeg
step. Run inside the `f5tts` conda env (activate.d sets LD_LIBRARY_PATH for
torchcodec). The F5TTS_v1_Base model auto-downloads from HF on first run.
"""
import sys

import soundfile as sf
import torch

from f5_tts.api import F5TTS

REF_WAV = "/home/prada/code/danill/experiments/danil/reference/danill_ref9.wav"
REF_TEXT = (
    "This is my natural speaking voice, calm, clear and steady. "
    "As I read these few lines aloud today, I speak at an even pace with"
)
SMOKE_TEXT = (
    "Hello — this is a quick zero-shot test of the cloned voice, "
    "reading a couple of natural English sentences aloud."
)
OUT_RAW = "/home/prada/code/danill/experiments/_tmp_f5_raw.wav"


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[f5] torch {torch.__version__} | device={device}")

    f5 = F5TTS(model="F5TTS_v1_Base", device=device)
    print("[f5] model loaded")

    print("[f5] synthesizing smoke sentence (zero-shot: ref clip + ref text)...")
    wav, sr, _spec = f5.infer(
        ref_file=REF_WAV,
        ref_text=REF_TEXT,
        gen_text=SMOKE_TEXT,
        nfe_step=32,
        cfg_strength=2.0,
        speed=1.0,
        seed=42,
    )

    sf.write(OUT_RAW, wav, sr)
    dur = len(wav) / sr
    print(f"[f5] OK wrote {OUT_RAW} | sample_rate={sr} | duration={dur:.2f}s")
    print(f"SAMPLE_RATE={sr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
