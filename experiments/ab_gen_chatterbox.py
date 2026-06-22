"""Batch-generate the shared A/B sentences on Chatterbox (zero-shot).

Run in the `chatterbox` conda env. Mirrors smoke_chatterbox.py (audio_prompt_path=ref9,
save via soundfile NOT torchaudio.save). Writes raw wavs at model.sr (24 kHz).
"""
import os
import json
import numpy as np
import soundfile as sf
import torch
from chatterbox.tts import ChatterboxTTS

ROOT = "/home/prada/code/danill"
REF = f"{ROOT}/experiments/danil/reference/danill_ref9.wav"
S = json.loads(open(f"{ROOT}/experiments/ab_sentences.json").read())
OUT = f"{ROOT}/experiments/danil/ab/chatterbox"
os.makedirs(OUT, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)
for i, text in enumerate(S, 1):
    wav = model.generate(text, audio_prompt_path=REF,
                         exaggeration=0.5, cfg_weight=0.5, temperature=0.8)
    audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
    sf.write(f"{OUT}/s{i:02d}.wav", audio, model.sr)
    print(f"s{i:02d} dur={len(audio) / model.sr:.2f}s")
print("CHATTERBOX_AB_DONE")
