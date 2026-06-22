#!/usr/bin/env python
"""XTTS-v2 zero-shot smoke test (workstream 2, engine 1).

Loads the local XTTS-v2 base model, conditions on the reference clip, and
synthesizes the smoke sentence. Writes raw 24 kHz output; loudness norm is a
separate ffmpeg step. Run inside the `xtts` conda env (so activate.d sets
LD_LIBRARY_PATH for torchcodec).
"""
import os
import sys

import torch

MODEL_DIR = "/home/prada/code/danill/third_party/xtts_v2_model"
REF_WAV = "/home/prada/code/danill/experiments/danil/reference/danill_ref9.wav"
OUT_RAW = "/home/prada/code/danill/experiments/_tmp_xtts_raw.wav"
SMOKE_TEXT = (
    "Hello — this is a quick zero-shot test of the cloned voice, "
    "reading a couple of natural English sentences aloud."
)

# --- torch>=2.6 weights_only gotcha -------------------------------------
# XTTS checkpoints pickle Coqui config dataclasses; allowlist them so
# torch.load(weights_only=True) (the new default) does not refuse to unpickle.
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals(
    [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]
)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[xtts] torch {torch.__version__} | device={device}")

    config = XttsConfig()
    config.load_json(os.path.join(MODEL_DIR, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, use_deepspeed=False)
    model.to(device)
    print("[xtts] model loaded")

    print("[xtts] computing conditioning latents from reference clip...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[REF_WAV]
    )

    print("[xtts] synthesizing smoke sentence (zero-shot)...")
    out = model.inference(
        SMOKE_TEXT,
        language="en",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7,
    )

    wav = torch.tensor(out["wav"]).unsqueeze(0)
    sr = getattr(model.config.audio, "output_sample_rate", 24000)

    import torchaudio

    torchaudio.save(OUT_RAW, wav, sr)
    dur = wav.shape[-1] / sr
    print(f"[xtts] OK wrote {OUT_RAW} | sample_rate={sr} | duration={dur:.2f}s")
    print(f"SAMPLE_RATE={sr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
