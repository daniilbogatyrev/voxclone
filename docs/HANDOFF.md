# voxclone — Handoff / State Guide

_Captures live project state + the per-engine install runbook that isn't obvious from the code, so a
fresh session (or teammate) can pick up. Deep research (fine-tune + zero-shot):
`docs/research-2026-05-25-finetune-zeroshot.md`._

> **Fresh-machine setup:** see `docs/SETUP.md` for the full clone → notebook-running guide. The per-engine
> env runbooks here (Workstream ② + GPT-SoVITS) are what SETUP.md links to for the non-obvious fixes.

> **Paths are repo-relative.** This doc uses `<repo>` for the repository root and `~/miniforge3` for the
> conda install; substitute your own. Never hardcode an absolute user path (a clone under any other
> user/home must work — `tests/test_no_hardcoded_paths.py` enforces this for runtime code).

## What this is
Personal, **non-commercial** project to clone one consenting person's voice and compare open-source TTS
engines. A clean `voxclone` library exposed three ways: CLI, FastAPI + vanilla-JS web app, and **one
guided Jupyter notebook** (`notebooks/voice_cloning_study.ipynb`, built by
`notebooks/build_voice_cloning_study.py`).

> **Public release:** the repo is published **without any voice audio or fine-tuned weights** (consent/
> privacy). The notebook ships in two builds from one source: `--audience public` (committed; clone-your-
> own-voice, no author audio) and `--audience professor` (ZIP-only; keeps the author's voice + embedded
> audio). The fine-tuned checkpoints + reference clip travel out of band in `daniil_starter_kit.zip`.

## Model decision (settled — do not re-litigate)
Fine-tune **XTTS-v2 + F5-TTS** (co-primary, native English); keep **GPT-SoVITS v2Pro** as a comparison
(zero-shot); **Chatterbox is a zero-shot baseline ONLY** (do NOT fine-tune it — unofficial/unproven for
English). Winner chosen by metric: speaker-similarity vs a real-vs-real ceiling + UTMOS naturalness + WER,
weighted **0.30 / 0.50 / 0.20**, plus a blind A/B.

## Status — fine-tunes shipped; F5 wins
- ✅ **Pipeline complete** (~344 GPU-free tests): capture/prep (`voxclone.capture.prompts` 814-sentence
  bank, SNR gate, 3–11 s clip cap, calibration + session cap; `prep/split.py` train/held-out/enrollment
  split), eval fairness (`eval/normalize.py` 16k/mono/−23 LUFS front end for ALL clips, ECAPA similarity,
  WER via Whisper `EnglishTextNormalizer` + DQ floor, ceiling-relative similarity, `EvalConfig`).
- ✅ **Four engines installed**, each its own conda env on the Blackwell stack (torch 2.11.0+cu128, sm_120) —
  see the **Workstream ② + GPT-SoVITS runbooks below** for the per-engine pins + NPP gotchas.
- ✅ **Fine-tunes trained + scored:** XTTS-v2 + F5-TTS fine-tuned on the consenting voice; the committed
  results live in `experiments/ab_eval_finetune_results.json`. **F5-TTS leads in both modes** (≈ 90 % of the
  real-vs-real ceiling) and **F5 zero-shot beats XTTS fine-tuned** — choosing the right engine matters more
  than fine-tuning the weaker one. Full write-up: `docs/finetune-vs-zeroshot-report.md`.
- ✅ **The studio notebook** auto-launches per-engine FastAPI servers from a thin `.venv` kernel and clones
  typed text live (zero-terminal). Backend: `voxclone.clone.studio`. See the 2026-06-02 clone-studio spec.

## Quantitative baseline (zero-shot, N=6 shared sentences/engine, all conditioned on the 9 s reference)

| rank | engine | sim (% of ceiling) | UTMOS (±σ) | WER | score |
|---|---|---|---|---|---|
| 1 | **F5-TTS** | 0.736 (89%) | 3.865 (±0.10) | 0.008 | **0.851** |
| 2 | XTTS-v2 | 0.545 (66%) | 3.716 (±0.13) | 0.000 | 0.769 |
| 3 | Chatterbox | 0.556 (67%) | 3.610 (±0.18) | 0.008 | 0.761 |
| 4 | GPT-SoVITS v2Pro | 0.440 (53%) | 3.630 (±0.22) | 0.008 | 0.721 |

(weights sim 0.30 / nat 0.50 / wer 0.20; ceiling 0.8284; real-voice UTMOS 3.295.) **F5-TTS clearly best
and most consistent; GPT-SoVITS last + most erratic.** Methodology + caveats (don't over-read absolutes):
- ECAPA (speechbrain) similarity to a held-out real centroid, as % of a real-vs-real ceiling; WhisperX
  `large-v3` WER (Whisper-normalized); **SpeechMOS `utmos22_strong`** UTMOS via `torch.hub`.
- The ceiling is two halves of one same-session take, so similarity is reference-biased but EQUALLY across
  engines → the cross-engine RANKING is fair, the absolute %s optimistic.
- **UTMOS rates every clone ABOVE the real recording** — a clean-audio quirk; read it as "polished," not
  "more human than human."
- Rerunnable scripts: `experiments/{eval_zeroshot,ab_eval,ab_eval_finetune,ab_gen_*}.py` + `ab_sentences.json`;
  numbers in `experiments/{eval_zeroshot,ab_eval,ab_eval_finetune}_results.json`. (Generated audio is gitignored.)

## ⚠️ GPT-SoVITS runtime — installed, with TWO non-obvious fixes (READ THIS FIRST)
Install: `<repo>/third_party/GPT-SoVITS` (gitignored), own conda env **`gptsovits`** (Python 3.11), built
with `bash install.sh --device CU128 --source HF`. `install.sh` left two things broken that we fixed:

1. **torchaudio CUDA mismatch** — it pulled `torchaudio` for CUDA 13 while torch is cu128.
   Fix: `pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps torchaudio` → `2.11.0+cu128`.
2. **torchcodec can't find NVIDIA NPP** — THE gotcha. torchcodec's core links `libnppicc.so.12`
   (NPP); `--index-url cu128` never installed NPP, and even once installed, torchcodec's RPATH
   doesn't include it. **Two-part fix, both required:**
   - `pip install nvidia-npp-cu12==12.3.3.100` (the CUDA-12.8-matched build; `12.4.x` is for 12.9), AND
   - launch the server with `LD_LIBRARY_PATH` including `$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/npp/lib`.
   (Env ffmpeg was also downgraded 8.1.1 → **7.1.1**; not the real cause, but 7 works and is what's installed.)

> **⚠️ The `LD_LIBRARY_PATH` is NOT persisted** by default — the server only decodes audio when launched
> with it. The studio injects it automatically; manual launches need it (or an activate.d hook).

## Start the api_v2 server (the WORKING command)
```bash
~/miniforge3/bin/conda run --no-capture-output -n gptsovits bash -c '
  cd <repo>/third_party/GPT-SoVITS
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"
  export TERM=xterm
  exec python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml'
```
Loads **v2 base** by default. Switch to **v2Pro** at runtime (no restart needed):
```
GET http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1v3.ckpt
GET http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth
```
Liveness: any HTTP response on `http://127.0.0.1:9880/` (a 404 on `/` means it's up).

## Reliable zero-shot synthesis recipe (English) — IMPORTANT
The v2 base model **collapses to silence** on text split into short fragments (premature EOS — a known
English-instability of this Chinese-origin model). To synthesize ANY text reliably on **v2Pro**:
- `text_split_method`: **`cut0`** for a single sentence (≤~25 words); **`cut4`** for multi-sentence.
  **Never `cut5`** for English (it splits on every comma → fragments collapse).
- Chunk long text into 1–2 sentences per call (externally); stitch with ~0.3 s gaps.
- `temperature` **0.6–0.8** (default 1.0), `repetition_penalty` 1.35–1.5, `top_k` 5–15.
- Normalize numbers/abbreviations/symbols before sending. v2Pro **requires** a non-empty
  `prompt_text` = the exact transcript of the reference clip.
- Output is 32 kHz float; normalize to −16 LUFS (`ffmpeg loudnorm`) for comfortable listening.

## GPT-SoVITS api_v2 reference (verified from the repo source)
- `POST /tts` JSON body → returns **WAV bytes**. Fields: `text`, `text_lang`, `ref_audio_path`,
  `prompt_text`, `prompt_lang`, `text_split_method` (use **cut0/cut4**, not cut5), `batch_size:1`,
  `media_type:"wav"`, `streaming_mode:false` (+ optional `top_k/top_p/temperature/repetition_penalty/seed/speed_factor`).
- `GET /set_gpt_weights?weights_path=...` and `GET /set_sovits_weights?weights_path=...` swap weights.
- v2Pro output sample rate: **32 kHz** (adapter returns float32). Base models ~3.8 GB; full install ~8 GB.

## Full fine-tune (GPT-SoVITS path) — researched, used as a comparison
Pipeline (repo file:line cites in `docs/research-2026-05-25-finetune-zeroshot.md`):
record → slice → Whisper ASR (`.list`) → 4 prep steps (`1-get-text`, `2-get-hubert-wav32k`,
**`2-get-sv`** [v2Pro-only speaker-verification embeddings], `3-get-semantic`) →
`s2_train.py` (SoVITS, ~8 ep) → `s1_train.py` (GPT, ~15 ep) → load via `set_*_weights`. (LoRA is v3/v4 only.)
- **Data:** ~45–60 min / 300–700 clips; minutes–2 h on a 24 GB+ GPU (fp16, batch ~12).
- **English ceiling:** BERT features are zero-padded for English + Chinese HuBERT → prosodic flatness that
  fine-tuning improves but can't remove. This is exactly why the plan also fine-tunes XTTS-v2 + F5-TTS.
- The XTTS/F5 fine-tune runbook (the co-primary path that actually won) is `docs/RUNBOOK_finetune.md`.

## Dev quickstart
```bash
cd <repo>
uv sync --extra dev
uv run --extra dev pytest -q            # ~344 tests, GPU-free
```
- pytest is a **dev extra** — `uv run pytest` fails; use `uv run --extra dev pytest`.
- Always use the project interpreter (`<repo>/.venv/bin/python`), never bare `python` (the shell may
  default to an unrelated conda env). CLI scripts live in **`src/scripts/`** (entry points
  `voxclone-prep`/`train`/`eval`/`serve`).

## Workstream ② — XTTS-v2 / F5-TTS / Chatterbox (each its own conda env, validated)
All on the Blackwell stack **torch 2.11.0+cu128** (sm_120), producing valid audio at **24 kHz**
(GPT-SoVITS is 32 kHz — resample for a strict A/B). Smoke scripts: `experiments/smoke_{xtts,f5,chatterbox}.py`.
**Do NOT touch the `gptsovits` env or repo when working in these.**

> **Cross-cutting gotcha — torchcodec + `libnppicc.so.12`:** torch cu128 wheels don't bundle NVIDIA NPP, but
> `torchcodec` (and torch 2.11's `torchaudio.save`) link `libnppicc.so.12`. Fix per env: `pip install
> nvidia-npp-cu12` + a `$CONDA_PREFIX/etc/conda/activate.d/zz_nvidia_libs.sh` that prepends each
> `site-packages/nvidia/*/lib` to `LD_LIBRARY_PATH` on activate. (Same root cause as the GPT-SoVITS NPP issue.)

- **XTTS-v2 — env `xtts`** (PRIMARY): coqui-tts 0.27.5 (idiap fork, imports as `TTS`), **transformers
  4.57.6** (pin ≤4.57.6; 5.x removes `isin_mps_friendly`, idiap #558), torchcodec 0.11.1+cu128. Install
  torch FIRST (BYO-torch). Loading needs `torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig,
  XttsArgs, BaseDatasetConfig])` (torch≥2.6 `weights_only`). Model in `<repo>/third_party/xtts_v2_model/` (HF
  `coqui/XTTS-v2`). Run: `conda activate xtts && COQUI_TOS_AGREED=1 python experiments/smoke_xtts.py`.
- **F5-TTS — env `f5tts`** (CO-PRIMARY): f5-tts 1.1.20 (`F5TTS_v1_Base` + Vocos), transformers 5.9.0,
  torchcodec 0.11.1+cu128. Flexible torch pin (install torch first, then `pip install f5-tts`). Zero-shot
  conditions on a reference clip **+ its transcript**. Model auto-downloads. Run: `conda activate f5tts &&
  python experiments/smoke_f5.py`.
- **Chatterbox — env `chatterbox`** (baseline): chatterbox-tts 0.1.7, transformers 5.2.0, numpy 1.26.4,
  **no torchcodec**. Package **hard-pins torch==2.6.0** (unusable on sm_120) → install torch 2.11.0+cu128
  first, then `pip install --no-deps chatterbox-tts`, then its other deps minus torch/torchaudio (librosa,
  resemble-perth, conformer, diffusers, omegaconf, pyloudnorm). **Save audio with soundfile, NOT
  `torchaudio.save`** (which needs torchcodec here). Model auto-downloads. Run: `conda activate chatterbox &&
  python experiments/smoke_chatterbox.py`.
