"""Generate the shared A/B sentences on the FINE-TUNED XTTS model (runs/xtts).

Same ref clip + params as ab_gen_xtts.py (the zero-shot generator), but loads the
fine-tuned checkpoint (best_model.pth + the run's config.json). Run in `xtts`.
"""
import os
import glob
import json
import soundfile as sf
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

ROOT = "/home/prada/code/danill"
RUN = sorted(glob.glob(f"{ROOT}/runs/xtts/xtts_finetune_danil-*"))[-1]
REF = f"{ROOT}/experiments/danil/reference/danill_ref9.wav"
S = json.loads(open(f"{ROOT}/experiments/ab_sentences.json").read())
OUT = f"{ROOT}/experiments/danil/ab/xtts_finetuned"
os.makedirs(OUT, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
config = XttsConfig()
config.load_json(RUN + "/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=RUN + "/best_model.pth",
                      vocab_path=f"{ROOT}/third_party/xtts_v2_model/vocab.json",
                      use_deepspeed=False)
model.to(device)
gpt_cond, spk = model.get_conditioning_latents(audio_path=[REF])
sr = getattr(model.config.audio, "output_sample_rate", 24000)
for i, text in enumerate(S, 1):
    out = model.inference(text, language="en", gpt_cond_latent=gpt_cond,
                          speaker_embedding=spk, temperature=0.7)
    sf.write(f"{OUT}/s{i:02d}.wav", out["wav"], sr)
    print(f"s{i:02d}")
print("XTTS_FT_AB_DONE")
