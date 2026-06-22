# RUNBOOK — fine-tune the English voice clone (run-ready)

Status: **wired + unit-tested GPU-FREE, NEVER run on real hardware/data.** The gating
item is the dataset: nothing downstream can run until the ~50-min English corpus is
recorded and prepped. This runbook is the exact runtime path so a future GPU session can
execute the bake-off once that recording exists.

Primaries to fine-tune: **XTTS-v2** and **F5-TTS** (co-primary). **GPT-SoVITS** is
**comparison-only** and its fine-tune path is a STUB (see Known Limitations §1).
**Chatterbox** is zero-shot only — never fine-tuned, not in `voxclone.train.TRAIN_ENGINES`.

Code references (read before running):
`src/scripts/{train,eval,serve,prep}.py`, `src/voxclone/train/{xtts,f5,gptsovits,xtts_recipe}.py`,
`src/voxclone/prep/{pipeline,split,manifest}.py`, `src/voxclone/capture/prompts.py`,
`src/voxclone/common/config.py` (EvalConfig weights), `src/voxclone/common/registry.py`.

---

## 0. GATING human step — record + prep the dataset

Nothing below runs without this. **This is a human capture session, not a script.**

### 0a. Record ~50 min of clean English

Read the **Plan 05 prompts** from `voxclone.capture.prompts` (module
`src/voxclone/capture/prompts.py`). `PROMPTS` is a dict of 4 categories — `harvard`
(Harvard sentences from `capture/data/harvard_sentences.txt`), `expressive`,
`conversational`, `technical`. Use `get_prompts(category)` or `all_prompts()` (returns
`[(category, text), ...]`) to drive the read.

Recording requirements (these gate the prep validator, see §0b):
- **24 kHz mono** WAV. (F5's `prepare_csv_wavs.py` does NOT resample — clips must already
  be the target rate; XTTS internally works at 22.05 kHz and the recipe outputs 24 kHz.)
  NOTE: `configs/default.yaml` ships `target_sample_rate: 48000` / `capture.sample_rate:
  48000`. For a 24 kHz dataset, **set `prep.target_sample_rate: 24000` (and
  `capture.sample_rate: 24000`) in the config you pass to `voxclone-prep`** — do not run
  prep with the 48 kHz default. (Resampling to 24 kHz is also fine if you captured higher,
  since `run_prep` resamples to `target_sample_rate`; the binding constraint is that the
  manifest's wavs are 24 kHz mono before F5 prep.)
- **SNR ≥ 30 dB.** The prep validator (`ValidationConfig.min_snr_db = 30.0` in
  `common/config.py`) quarantines clips below this. Record in a quiet room, consistent
  mic distance.
- Per-clip duration roughly 3–11 s (the VAD segmenter + `ValidationConfig` duration window
  govern what is kept; clips outside the window or with low transcript confidence are
  quarantined). Aim for ~50 min of KEPT audio after quarantine, not 50 min of raw take.
- Filename stems should carry the category prefix (`harvard_*`, `expressive_*`, etc.):
  `run_prep` derives `ClipRecord.category` from the stem before the first `_`, and the
  held-out split spans categories.

Drop the raw master WAVs into a `raw/` dir.

### 0b. Prep -> manifest

```
voxclone-prep --raw <raw_dir> --out data/clips \
  --config <prep_config.yaml> \
  --manifest data/manifest.jsonl --transcripts data/transcripts.csv
```

`run_prep` (`src/voxclone/prep/pipeline.py`) resamples to `target_sample_rate`, denoises,
VAD-segments, peak-normalizes, transcribes (Whisper `large-v3`), validates each clip
(duration / clipped-fraction / transcript-confidence / **SNR ≥ 30**), and writes
`data/manifest.jsonl` (kept clips) plus a quarantine report. Inspect the quarantine report;
if too much was dropped, re-record. Whisper + VAD are GPU steps — run inside a GPU env.

### 0c. Split -> train / held-out / enrollment

There is **no split CLI** — `voxclone.prep.split` exposes library functions. Run a tiny
driver (inside the project `.venv`) over the manifest:

```python
from voxclone.prep.manifest import read_manifest, write_manifest
from voxclone.prep.split import split_dataset, write_heldout_tsv, write_enrollment

recs = read_manifest("data/manifest.jsonl")
s = split_dataset(recs, n_heldout=20, n_enrollment=4, seed=0)   # deterministic
write_manifest("data/train.jsonl", s.train)                     # -> train fine-tune input
write_heldout_tsv("data/held_out.tsv", s.held_out)              # text<TAB>real_clip  (eval --held-out)
write_enrollment("data/enrollment.tsv", s.enrollment)           # path<TAB>text       (synth refs)
```

`split_dataset` carves: enrollment = cleanest in-window clips (6–10 s) for synth
conditioning; held-out ≈ 20 category-spanning, de-duplicated clips for eval; train = the
rest, with near-duplicate texts kept off the train/held-out boundary. **`data/train.jsonl`
is the manifest you pass to `voxclone-train` for every engine.** Pick one enrollment clip +
its verbatim transcript as the serve/eval `--reference` (F5 needs both the clip and the
transcript as `prompt_text`).

---

## 1. Per-engine fine-tune (each in its OWN conda env)

`voxclone-train` (`src/scripts/train.py`) dispatches `--engine` through
`voxclone.train.TRAIN_ENGINES`, constructs the trainer with only its own roots/conda-env
args, and calls `.train(manifest, out, config["train"])` with the YAML's `train:`
sub-mapping. `--out` defaults to `runs/<engine>`; **its basename is load-bearing** —
`runs/xtts` basename `xtts` == engine key == registry key == served model name. Do not
rename it.

### 1a. XTTS-v2 — env `xtts` (coqui-tts 0.27.5, transformers ≤ 4.57.6)

```
voxclone-train --engine xtts \
  --manifest data/train.jsonl \
  --config configs/xtts.yaml \
  --out runs/xtts \
  --xtts-root third_party/xtts_v2_model
```

- `XTTSTrainer` (`train/xtts.py`) writes the two coqui-format CSVs
  (`metadata_train.csv` / `metadata_eval.csv`; that eval split is XTTS's **in-training loss
  split**, NOT the project held-out set) then shells out via `runner` to:
  `conda run -n xtts python -m voxclone.train.xtts_recipe --model_dir <xtts-root> ...`
- `--xtts-root` must point at a **full XTTS-v2 model dir** containing `config.json`,
  `model.pth`, `vocab.json`, **`dvae.pth`**, **`mel_stats.pth`**. The HF `coqui/XTTS-v2`
  snapshot ships all five; `xtts_recipe.main` raises `FileNotFoundError` if any is missing.
- The recipe uses torch ≥ 2.6 `torch.serialization.add_safe_globals(...)` to load the
  checkpoint under `weights_only`. Keep `transformers ≤ 4.57.6` in the `xtts` env.
- Small-dataset `train:` overrides (else the recipe's big-corpus defaults are wrong):
  `epochs` 6–15 (default 10), `batch_size` 3–6 (default 4), `grad_accum` 1–4 (default 2),
  `lr` 5e-6, `mixed_precision: true`, `max_wav_length 255995`, `max_text_length 200`.
  XTTS silently **drops** (not truncates) clips/texts longer than those caps.
- Output: checkpoints + logs under `runs/xtts/`.

### 1b. F5-TTS — env `f5tts` (f5-tts 1.1.20)

```
voxclone-train --engine f5 \
  --manifest data/train.jsonl \
  --config configs/f5.yaml \
  --out runs/f5
```

- **PREREQUISITE (one-time, do before the first finetune):** the pretrained vocab must
  exist at `<f5_data_root>/Emilia_ZH_EN_pinyin/vocab.txt`. `prepare_csv_wavs.py` run
  WITHOUT `--pretrain` (the trainer runs it in finetune mode) copies that pretrained
  `vocab.txt` for the new dataset; if it is absent, fetch it first, e.g.:

  ```python
  from cached_path import cached_path
  cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt")
  ```
  (or an equivalent `huggingface_hub` download), and place it under the f5 data root's
  `Emilia_ZH_EN_pinyin/` dir for the installed f5-tts.

- `F5Trainer` (`train/f5.py`) runs two GPU steps via `runner`, both inside `conda run -n
  f5tts`:
  - (A) `python <prepare_csv_wavs.py> <metadata.csv> <data/<dataset_name>>` — builds the
    Arrow dataset; finetune mode (no `--pretrain`) copies the pretrained vocab.
  - (B) `accelerate launch --mixed_precision=bf16 -m f5_tts.train.finetune_cli`
    with flags: `--exp_name F5TTS_v1_Base --dataset_name danil --finetune
    --tokenizer pinyin --learning_rate 1e-5 --batch_size_per_gpu 3200 --batch_size_type
    frame` (plus `--epochs`, `--max_samples`, `--grad_accumulation_steps`,
    `--num_warmup_updates`, `--save_per_updates`, `--last_per_updates`,
    `--keep_last_n_checkpoints 3`, `--logger tensorboard --log_samples`).
    `--tokenizer pinyin` is intentional even for English — it matches the pretrained
    embedding. (Default `dataset_name` in the code is `danil_pinyin`; pass
    `--dataset-name`/ctor arg if you want a different name — keep it consistent with the
    vocab dir.)
- metadata CSV: header EXACTLY `audio_file|text`, col1 = ABSOLUTE 24 kHz mono wav path
  (the trainer rejects non-absolute paths). prepare does NOT resample.
- **Checkpoint location:** there is **NO `--output_dir`/`--save_dir`** flag — F5 derives
  the checkpoint dir solely from `--dataset_name` and writes
  `<f5_pkg>/../../ckpts/{dataset_name}/model_last.pt`. **IMPORTANT (gap):** the current
  `F5Trainer.train` returns `TrainResult(checkpoint_dir=runs/f5)` but does **NOT** copy
  `model_last.pt` from the f5 `ckpts/{dataset_name}/` dir into `runs/f5/`. Until that copy
  is added, **manually copy `ckpts/{dataset_name}/model_last.pt` (and the dataset
  `vocab.txt`) into `runs/f5/` after training** so eval/serve resolve it via the registry.
  (Fixing the trainer to copy is the intended follow-up; this runbook documents what the
  code does today.)

### 1c. GPT-SoVITS — env `gptsovits` (comparison-only — STUB)

The GPT-SoVITS fine-tune is **not run-ready** — see Known Limitation §1. Do NOT attempt to
fine-tune it for the bake-off; include GPT-SoVITS only as a **zero-shot** candidate in eval
(§2). If/when the config-file wiring is done, it would run as:

```
voxclone-train --engine gptsovits --manifest data/train.jsonl \
  --config configs/gptsovits.yaml --out runs/gptsovits \
  --gptsovits-root third_party/GPT-SoVITS
```

but the trainer currently builds **placeholder FLAG args** that the pinned repo's
`s2_train.py` / `s1_train.py` do not accept (they are config-FILE driven). It will not run
as-is.

---

## 2. Evaluate the bake-off

```
voxclone-eval \
  --candidates xtts:finetuned=runs/xtts f5:finetuned=runs/f5 \
               gptsovits:zeroshot=base chatterbox:zeroshot=base \
  --reference data/enrollment_clip.wav \
  --held-out data/held_out.tsv \
  --report reports/eval.md \
  --register runs \
  --device cuda
```

- Each `--candidates` token is `engine:label=checkpoint` (`engine` selects the
  `SYNTH_ENGINES` class; `label` = `finetuned` vs `zeroshot`). The candidate KEY written to
  the registry is `<engine>_<label>` (e.g. `xtts_finetuned`).
- `--reference` = an enrollment clip (the cloned voice reference); `--held-out` = the
  `text<TAB>real_clip` TSV from §0c. Both are **required** by `scripts/eval.py` (the brief's
  shorthand omitted them — they are mandatory).
- Scoring weights come from `EvalConfig` (`common/config.py`):
  **similarity 0.30 / naturalness 0.50 / wer 0.20**, with WER disqualification threshold
  `0.20`. Pass `--config` to override.
- With `--register runs`, scored candidates are written to the `ModelRegistry`
  (`runs/registry.json`) under their compound keys with metrics incl. `score`, so `serve`
  can later resolve the best checkpoint **per short engine key**. (Note: the registry's
  `best_checkpoint(model)` keys on the SHORT engine name `xtts`/`f5`/...; the eval candidate
  key is the COMPOUND `xtts_finetuned`. Confirm the engine vs compound key resolves the way
  you expect before relying on serve auto-resolution — see Known Limitation §3.)
- Expected ordering from prior zero-shot A/B (NOT a substitute for the real run): F5 #1,
  GPT-SoVITS last. The fine-tuned XTTS/F5 numbers are unknown until this runs on GPU.

---

## 3. Serve the winner

```
voxclone-serve --runs runs --reference data/enrollment_clip.wav \
  --host 127.0.0.1 --port 8000
```

- `scripts/serve.py` builds a provider over `ModelRegistry(runs)` and routes **per-request
  by `req.model`** (the `SynthRequest.model` field; default `"f5"`) to
  `SYNTH_ENGINES[model](checkpoint=reg.best_checkpoint(model))`. **There is NO
  `--model` CLI flag** (the brief's `voxclone-serve --model <engine>` is inaccurate); the
  served model is chosen by the POST request body, and the server resolves that model's
  best-scoring registered checkpoint. Unknown / unregistered model -> 404.

---

## Known limitations (state plainly)

1. **GPT-SoVITS fine-tune is a STUB.** `GPTSoVITSTrainer` (`train/gptsovits.py`) invokes
   `s2_train.py` / `s1_train.py` with `--list/--exp/--epochs` **FLAG** args. The pinned
   GPT-SoVITS repo's training is **config-FILE driven**: `s2_train.py -c <config>.json` and
   `s1_train.py -c <config>.yaml` (see `third_party/GPT-SoVITS/GPT_SoVITS/configs/`). The
   current flags will NOT run. This path needs **generated config files (JSON for s2, YAML
   for s1) matching the pinned version** before any real run. GPT-SoVITS is
   **comparison-only** (XTTS + F5 are the primaries), so this is acceptable for the bake-off
   — use GPT-SoVITS zero-shot in eval, not fine-tuned.

2. **No GPU run has ever executed.** Everything above is wired and unit-tested **GPU-FREE**
   (heavy work mocked via injected seams: `runner` / `generate_fn` / `model_factory` /
   `httpx`). None of it has been validated on real hardware or real data. Treat the first
   GPU run as a bring-up: expect to debug dataset rates, missing model files
   (`dvae.pth`/`mel_stats.pth`, F5 `vocab.txt`), conda-env package pins, and OOM/batch sizes.

3. **F5 checkpoint is not auto-collected; registry key shapes differ.** (a) `F5Trainer`
   does not copy `model_last.pt` into `runs/f5/` — copy it manually until the trainer is
   fixed (§1b). (b) eval registers COMPOUND keys (`xtts_finetuned`) while `serve` /
   `ModelRegistry.best_checkpoint` resolve by SHORT engine key (`xtts`); verify the
   train -> registry -> serve naming actually lines up end-to-end on the first real run.

4. **Config sample-rate default is 48 kHz.** `configs/default.yaml` defaults to 48 kHz; the
   F5/XTTS path needs a 24 kHz mono manifest. Use a prep config with
   `prep.target_sample_rate: 24000` (§0a) — do not prep with the 48 kHz default.
