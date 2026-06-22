#!/usr/bin/env python
"""Chatterbox zero-shot smoke test (workstream 2, engine 3).

Chatterbox clones from a reference clip (audio_prompt_path) and generates the
target sentence. Writes raw output; loudness norm is a separate ffmpeg step.
Run inside the `chatterbox` conda env. The model auto-downloads from HF
(ResembleAI/chatterbox) on first run.
"""
import sys

import numpy as np
import soundfile as sf
import torch

from chatterbox.tts import ChatterboxTTS

REF_WAV = "/home/prada/code/danill/experiments/danil/reference/danill_ref9.wav"
SMOKE_TEXT = (
    "Hello — this is a quick zero-shot test of the cloned voice, "
    "reading a couple of natural English sentences aloud."
)
OUT_RAW = "/home/prada/code/danill/experiments/_tmp_chatterbox_raw.wav"


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[chatterbox] torch {torch.__version__} | device={device}")

    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"[chatterbox] model loaded | sr={model.sr}")

    print("[chatterbox] synthesizing smoke sentence (zero-shot clone)...")
    wav = model.generate(
        SMOKE_TEXT,
        audio_prompt_path=REF_WAV,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    )

    sr = model.sr
    # generate() returns a torch tensor shaped (1, n) or (n,). Save via
    # soundfile (1-D float32) — torchaudio.save in torch 2.11 routes through
    # torchcodec, which is intentionally not installed in this env.
    audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
    sf.write(OUT_RAW, audio, sr)
    dur = len(audio) / sr
    print(f"[chatterbox] OK wrote {OUT_RAW} | sample_rate={sr} | duration={dur:.2f}s")
    print(f"SAMPLE_RATE={sr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
