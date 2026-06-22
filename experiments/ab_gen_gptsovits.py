"""Batch-generate the shared A/B sentences on GPT-SoVITS v2Pro (zero-shot).

Run in the .venv (talks HTTP to the already-running api_v2 server on :9880).
Writes raw 32 kHz wavs; the eval's normalize_for_eval handles loudness/SR fairness.
"""
import json
from pathlib import Path
import numpy as np
from voxclone.synth.gptsovits import GPTSoVITSSynth
from voxclone.common.audio import save_audio

ROOT = Path("/home/prada/code/danill")
REF = str(ROOT / "experiments/danil/reference/danill_ref9.wav")
PROMPT = ("This is my natural speaking voice, calm, clear and steady. As I read these "
          "few lines aloud today, I speak at an even pace with")
S = json.loads((ROOT / "experiments/ab_sentences.json").read_text())
OUT = ROOT / "experiments/danil/ab/gptsovits_v2pro"
OUT.mkdir(parents=True, exist_ok=True)

synth = GPTSoVITSSynth(checkpoint="v2Pro")
for i, text in enumerate(S, 1):
    audio, sr = synth.synthesize(text, REF, {
        "prompt_text": PROMPT, "text_split_method": "cut0",
        "temperature": 0.7, "top_k": 15, "prompt_lang": "en", "text_lang": "en",
    })
    dur, peak = len(audio) / sr, float(np.max(np.abs(audio)))
    save_audio(OUT / f"s{i:02d}.wav", audio, sr)
    flag = "COLLAPSE" if dur < 2.0 or peak < 0.02 else "ok"
    print(f"s{i:02d} dur={dur:.2f}s peak={peak:.3f} {flag}")
print("GPTSOVITS_AB_DONE")
