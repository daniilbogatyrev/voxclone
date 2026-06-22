# Research — Fine-tuning & Zero-shot (GPT-SoVITS v2Pro, English)

_2026-05-25. Synthesized from a 3-agent research pass grounded in the cloned repo
(`third_party/GPT-SoVITS`, with file:line cites), official benchmarks, and the web. Companion to
`docs/HANDOFF.md` (which has the actionable runtime state)._

## 1. Zero-shot — can it synthesize any text? Yes, with guardrails.
The text→semantic (GPT) stage emits tokens until EOS. English is undertrained in this model, so:
- **Short fragments** (e.g. `cut5` splitting on every comma) → premature EOS → near-silence
  (matches GitHub issue #1992). This is what we hit on the v2 base today.
- **Very long unbroken text** → EOS never fires → runs to the 1500-token cap → garbled drift.

**Reliable recipe (v2Pro):**
- Split: **`cut0`** for one sentence (≤~25 words), **`cut4`** for multi-sentence (English `.`, digit-aware). **Avoid `cut5`.**
- Chunk long text to 1–2 sentences/call (hard cap 510 chars); stitch with ~0.3 s gaps.
- `temperature` 0.6–0.8 (default 1.0), `repetition_penalty` 1.35–1.5, `top_k` 5–15, fixed `seed` once good.
- Pre-normalize numbers/abbreviations/symbols (the g2p frontend mishandles them).

## 2. Fine-tune pipeline for v2Pro (repo-grounded)
```
record → slice (tools/slice_audio.py) → Whisper ASR (.list: wav|spk|EN|text)
 → 1-get-text.py            (phonemes/BERT → logs/<exp>/2-name2text.txt)
 → 2-get-hubert-wav32k.py   (HuBERT feats + 32k wav)  +  2-get-sv.py  ← v2Pro-only (ERes2NetV2 SV embeddings → 7-sv_cn/)
 → 3-get-semantic.py        (VQ semantic tokens → 6-name2semantic.tsv)
 → s2_train.py  (SoVITS, config GPT_SoVITS/configs/s2v2Pro.json, ~8 epochs)   → SoVITS_weights_v2Pro/<exp>_eN_sM.pth
 → s1_train.py  (GPT,    config GPT_SoVITS/configs/s1longer-v2.yaml, ~15 ep)  → GPT_weights_v2Pro/<exp>-eN.ckpt
 → serve: GET /set_sovits_weights + /set_gpt_weights (auto-detects v2Pro from the b"05" header byte)
```
- **v2Pro-specific:** the `2-get-sv` step (speaker-verification embedding, model already on disk at
  `pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt`) — this is v2Pro's quality edge. The
  `7-sv_cn/` dir is **mandatory** for v2Pro training (asserted in `module/data_utils.py`).
- **Full fine-tune only** for v2Pro (LoRA is v3/v4-only; branch at `webui.py`). Text submodules train
  at 0.4× LR to protect linguistics.
- **Compute (RTX 5090, 24 GB):** comfortable at fp16, batch ~12; minutes–~2 h depending on dataset.
  Don't run SoVITS+GPT simultaneously; close the inference server during training (VRAM).
- **Pitfalls:** overfitting (cap SoVITS ~8 ep, GPT ~15; webui hard-caps 25); clips must be 0.6–54 s
  or they're silently dropped; min ~100 clips (loops below that); run `2-get-sv` with version=v2Pro
  or training crashes; transcript accuracy is the #1 lever.

## 3. Data plan for a single-speaker English clone
The project design already targets **45–60 min / 300–700 clips
/ 3–15 s**, Whisper large-v3 + WhisperX confidence, denoise (light), VAD, peak-norm −1 dBFS,
quarantine. That's **above** best-practice minimums (5–30 min) — appropriate, and good given the
English weakness. Content mix (per 50 min): ~20 min phonetically-balanced (Harvard/IEEE sentences),
~12–15 min expressive (questions/exclamations/instructional/skeptical/casual), ~8–10 min freeform
conversational, ~5–8 min technical/numbers. **Consistency across sessions is critical** (same
room/mic/distance/gain; calibration sentence each session; ≤20 min speech/session to avoid fatigue
drift). SNR ≥ 30 dB.

**Gaps found in our prep vs best practice (worth fixing before a big recording session):**
- `capture/prompts.py` seeds only ~15 Harvard sentences → expand to 200–300 (20–30 full IEEE lists).
- Expressive prompts ~15 → 50–100, categorized.
- `validate.py` checks clipping but no explicit **SNR** measurement → add one.
- No documented **session-consistency** guidance → add to the recorder/CLI.

## 4. Zero-shot vs fine-tuned — expected gains & the English ceiling
Official zero-shot benchmarks (SeedTTS, Chinese testset — English differs in absolute but similar trend):
speaker similarity (SIM) **v2 ≈ 0.55 → v2Pro ≈ 0.71–0.74**, ground-truth ≈ 0.75; WER v2Pro ≈ 0.015.

Fine-tuning on one speaker adds: **consistency** of identity across varied text, **stability** on long
text (esp. with DPO — less babysitting), and learned **pacing/pauses/breath** (the biggest practical
win). It does **not** fix number/abbrev handling (text frontend) or the structural English ceiling.

**The English ceiling (key caveat):** GPT-SoVITS is Chinese-origin — for English, BERT context
features are **zero-padded** (no syntactic signal → flatter prosody, weak question-vs-statement
intonation) and the speech tokenizer (HuBERT) is Chinese-trained. Fine-tuning improves timbre/pacing
a lot but **cannot remove** this. Fine-tuned v2Pro English is great for narration/audiobook, but a
native-English model (XTTS-v2, ElevenLabs-class) will sound more natural — hence the plan's
metric-picked multi-model comparison.

**When to fine-tune:** want consistent identity + long-form + max similarity, and have ≥5 (ideally
15–30+) min clean data. Diminishing returns ~15–30 min (English ceiling binds); overfitting risk beyond.
Stay zero-shot for quick one-offs / prototyping.

_Sources: GPT-SoVITS wiki + DeepWiki, issue #1992, repo source (`text_segmentation_method.py`,
`s2_train.py`, `data_utils.py`, `webui.py`, `api_v2.py`), AllTalk/XTTS guide, Harvard sentences corpus._
