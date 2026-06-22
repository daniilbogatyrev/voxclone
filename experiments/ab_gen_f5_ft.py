"""Generate the shared A/B sentences on the FINE-TUNED F5 model (runs/f5).

Same ref clip + ref transcript + params as ab_gen_f5.py (the zero-shot generator), but
loads the fine-tuned checkpoint so the comparison is apples-to-apples. Run in `f5tts`.
"""
import os
import json
import soundfile as sf
import torch
from f5_tts.api import F5TTS

ROOT = "/home/prada/code/danill"
REF = f"{ROOT}/experiments/danil/reference/danill_ref9.wav"
REF_TEXT = ("This is my natural speaking voice, calm, clear and steady. "
            "As I read these few lines aloud today, I speak at an even pace with")
S = json.loads(open(f"{ROOT}/experiments/ab_sentences.json").read())
OUT = f"{ROOT}/experiments/danil/ab/f5_finetuned"
os.makedirs(OUT, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=f"{ROOT}/runs/f5/model_last.pt",
           vocab_file=f"{ROOT}/runs/f5/vocab.txt", device=device)
for i, text in enumerate(S, 1):
    wav, sr, _ = f5.infer(ref_file=REF, ref_text=REF_TEXT, gen_text=text,
                          nfe_step=32, cfg_strength=2.0, speed=1.0, seed=42)
    sf.write(f"{OUT}/s{i:02d}.wav", wav, sr)
    print(f"s{i:02d} dur={len(wav) / sr:.2f}s")
print("F5_FT_AB_DONE")
