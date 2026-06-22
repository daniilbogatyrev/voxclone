"""Batch-generate the shared A/B sentences on XTTS-v2 (zero-shot).

Run in the `xtts` conda env. Mirrors smoke_xtts.py (add_safe_globals, conditioning
latents from ref9). Writes raw 24 kHz wavs via torchaudio.save (xtts env has torchcodec).
"""
import os
import json
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

ROOT = "/home/prada/code/danill"
MODEL_DIR = f"{ROOT}/third_party/xtts_v2_model"
REF = f"{ROOT}/experiments/danil/reference/danill_ref9.wav"
S = json.loads(open(f"{ROOT}/experiments/ab_sentences.json").read())
OUT = f"{ROOT}/experiments/danil/ab/xtts_v2"
os.makedirs(OUT, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
config = XttsConfig()
config.load_json(os.path.join(MODEL_DIR, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, use_deepspeed=False)
model.to(device)
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[REF])
sr = getattr(model.config.audio, "output_sample_rate", 24000)

import torchaudio
for i, text in enumerate(S, 1):
    out = model.inference(text, language="en", gpt_cond_latent=gpt_cond_latent,
                          speaker_embedding=speaker_embedding, temperature=0.7)
    wav = torch.tensor(out["wav"]).unsqueeze(0)
    torchaudio.save(f"{OUT}/s{i:02d}.wav", wav, sr)
    print(f"s{i:02d} dur={wav.shape[-1] / sr:.2f}s")
print("XTTS_AB_DONE")
