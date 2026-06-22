# Fresh-clone setup → the voice-clone studio notebook

This is the end-to-end guide for someone who just cloned this repo on a fresh
machine and wants to reach **`notebooks/voice_cloning_study.ipynb` running** — the
zero-terminal "Daniil voice-clone studio" that auto-launches per-engine TTS
servers from inside the notebook kernel, then clones typed text into a voice.

> The notebook is a zero-terminal clone studio: one thin `.venv` kernel that
> auto-launches the per-engine TTS servers and clones typed text live.

Read this top to bottom — the steps are ordered. Expect a non-trivial amount of
work: a fresh clone is **far** from runnable (see the next section).

---

## What you get / what is NOT in the clone

The git repo contains the **code** (the `voxclone` library, the notebook
builder, scripts, docs) and nothing heavy. Everything needed to actually
generate audio is **gitignored** and absent on a fresh clone:

| Missing on a fresh clone | What it is | How you get it |
|---|---|---|
| `.venv/` | the project Python env + notebook kernel | step 1 (`uv sync`) |
| `~/miniforge3/envs/{xtts,f5tts,chatterbox,gptsovits}` | the 4 per-engine conda envs that actually run the TTS models | step 2 (hand-built; **no scripted installer exists**) |
| `third_party/xtts_v2_model/` | XTTS-v2 base model | step 3 (HF download) |
| `third_party/f5_tts_german/` | F5 German checkpoint (German toggle only) | step 3 (HF download) |
| `third_party/GPT-SoVITS/` | GPT-SoVITS checkout + ~8 GB pretrained models | step 3 (git clone + its installer) — **not used by the studio notebook** |
| `runs/f5/`, `runs/xtts/xtts_finetune_danil-*/` | the two **fine-tuned** voices' checkpoints | trained on private data; see `docs/RUNBOOK_finetune.md` |
| `experiments/danil/reference/danill_ref9.wav` | the private reference clip every voice clones | **private/consent-gated — not redistributable.** Use bring-your-own-voice instead (step 3) |
| registered Jupyter kernel `voxclone` | the kernelspec the notebook hard-pins | step 1 (`ipykernel install`) |

Two honest consequences up front:

1. **The two fine-tuned voices** (`f5_finetuned` — the default ⭐ — and
   `xtts_finetuned`) depend on `runs/` checkpoints trained on Daniil's private
   data. They cannot be downloaded. On a fresh clone you either re-train them
   (`docs/RUNBOOK_finetune.md`, needs the private dataset) or you skip them. The
   three **zero-shot** voices (`f5_zeroshot`, `xtts_zeroshot`, `chatterbox`)
   work with only the base models + conda envs + a reference clip.

2. **The reference clip is private.** As of this writing, `studio.REF_CLIP` is a
   hardcoded module constant and the committed notebook exposes no upload widget
   — see step 3 for the bring-your-own-voice reality and how to point the studio
   at your own clip.

---

## System prerequisites

This is a **GPU project**. The engine workers load models onto CUDA; a CPU-only
or non-NVIDIA machine cannot run any generation (the `.venv` pre-flight will
pass, but every `studio.say()` will fail in the worker).

- **OS:** Linux. (Built/verified on Zorin OS 18.1, Ubuntu-based, kernel 6.17.)
- **NVIDIA GPU + driver:** a CUDA GPU with a driver new enough for the cu128
  PyTorch wheels. The stack is pinned to **torch 2.11.0+cu128** for **Blackwell
  (sm_120)** — verified on an RTX 5090 Laptop (24 GB), driver 595.71.05. The
  wheels bundle the CUDA 12.8 runtime, so the *system* CUDA toolkit version does
  not matter — only the driver must be recent enough.
- **miniforge / conda** under `$HOME/miniforge3` (verified conda 25.11.0). The
  studio derives the env Python via `Path.home()/"miniforge3"/"envs"/<env>`, so
  conda **must** be miniforge at `~/miniforge3` (or you must patch `studio.py`).
- **uv** (verified 0.9.27). Install if missing:
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **ffmpeg** and **libsndfile** on the system (soundfile/torchaudio deps).
  Verified `ffmpeg 6.1.1` and `libsndfile.so.1` present.
- **VS Code** (recommended) — the committed `.vscode/settings.json` pins the
  notebook working directory to the repo root, which the studio needs.

---

## 1. Project venv (the notebook kernel)

The notebook kernel itself is thin: it imports only `voxclone.clone.studio`
(stdlib + numpy + soundfile). All heavy TTS libs live in the conda envs (step
2). From the repo root:

```bash
# install uv first if you don't have it (see prerequisites)
uv sync --extra dev
```

This builds `.venv/` with the runtime deps plus the `[dev]` extras the notebook
needs (`ipykernel>=6,<7`, `ipywidgets`, `nbformat`, `nbconvert`, `matplotlib`,
`ipython`, `pytest`, ...). Python `>=3.11` is required.

> Note (non-blocking): on the original box, uv picked the conda `soccer` env's
> Python 3.11.15 as the base interpreter (`.venv/pyvenv.cfg` shows
> `home = .../envs/soccer/bin`). This is **not** a dependency on `soccer` — on a
> fresh clone uv will pick any 3.11+ interpreter (or download one). The resulting
> `.venv` is a genuine isolated venv.

**Register the Jupyter kernel** the notebook hard-pins (kernelspec name
`voxclone`, display `VoxClone (.venv)`). Without this the notebook's pre-flight
cell raises a "wrong kernel" `SystemExit`:

```bash
.venv/bin/python -m ipykernel install --user --name voxclone --display-name "VoxClone (.venv)"
```

Quick sanity check (GPU-free):

```bash
.venv/bin/python -c "import voxclone, ipykernel, nbformat, ipywidgets, matplotlib; print('ok')"
```

---

## 2. Per-engine conda envs

The studio launches **one** engine worker at a time, in that engine's conda env,
by invoking the env's Python interpreter **directly by absolute path** (no
`conda activate`, no `conda run`). The notebook uses three envs (`f5tts`,
`xtts`, `chatterbox`); `gptsovits` is only for the older `scripts/serve_*.sh` /
A/B / train paths and is **not** used by the studio.

> There is **no scripted env-builder** in the repo. These are hand-built; the
> authoritative narrative source is `docs/HANDOFF.md` "Workstream ②" + its
> GPT-SoVITS section. The commands below reproduce the pins recon found on the
> working box. Treat exact patch versions as best-effort: the load-bearing
> constraints are torch-first-then-engine, the `transformers` pins, and the
> NVIDIA-NPP fix.

**All four envs are Python 3.11.15 / torch 2.11.0+cu128 / CUDA 12.8 / sm_120.**

### Cross-cutting gotcha: NVIDIA NPP (`libnppicc.so.12`)

torch's cu128 wheels do not bundle NVIDIA NPP, but `torchcodec` / torch 2.11
`torchaudio.save` link `libnppicc.so.12`. Fix per env: `pip install
nvidia-npp-cu12` **and** prepend each `$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/*/lib`
to `LD_LIBRARY_PATH`. On the working box `xtts`/`f5tts`/`chatterbox` solve this
**permanently** with an activate hook at
`$CONDA_PREFIX/etc/conda/activate.d/zz_nvidia_libs.sh` (it loops over every
`site-packages/nvidia/*/lib`). Create that hook in each env after install. The
studio also injects NPP into `LD_LIBRARY_PATH` itself via `worker_env()`, so the
studio works even without the hook — but the standalone `scripts/serve_*.sh`
rely on it.

### `xtts` (primary)

```bash
conda create -n xtts python=3.11 -y
conda run -n xtts pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n xtts pip install coqui-tts==0.27.5 coqui-tts-trainer==0.3.3 'transformers==4.57.6' torchcodec==0.11.1 nvidia-npp-cu12==12.4.1.87 fastapi==0.136.3 'uvicorn==0.48.0' httpx soundfile
```

- `transformers` **must** be `<=4.57.6` (5.x removes `isin_mps_friendly`).
- `coqui-tts` imports as `TTS`. Launch needs `COQUI_TOS_AGREED=1` (the studio
  sets this for you).

### `f5tts` (co-primary)

```bash
conda create -n f5tts python=3.11 -y
conda run -n f5tts pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n f5tts pip install f5-tts==1.1.20 torchcodec==0.11.1 nvidia-npp-cu12==12.4.1.87 fastapi==0.136.3 'uvicorn==0.48.0' httpx soundfile
```

The F5 base model auto-downloads (see the offline-cache note in Troubleshooting).

### `chatterbox` (zero-shot baseline)

The `chatterbox-tts` package hard-pins `torch==2.6.0` (unusable on sm_120), so
install torch first, then `--no-deps`:

```bash
conda create -n chatterbox python=3.11 -y
conda run -n chatterbox pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n chatterbox pip install --no-deps chatterbox-tts==0.1.7
conda run -n chatterbox pip install 'transformers==5.2.0' s3tokenizer==0.3.0 'numpy==1.26.4' nvidia-npp-cu12==12.4.1.87 fastapi==0.136.3 'uvicorn==0.48.0' httpx soundfile
```

- Note `numpy==1.26.4` here (not numpy 2.x). No torchcodec in this env; save
  audio via soundfile, not `torchaudio.save`. German uses
  `ChatterboxMultilingualTTS` (auto-downloads).

### `gptsovits` (comparison — NOT used by the studio notebook)

Only needed for `scripts/serve_engines.sh` / `scripts/serve_finetuned.sh` /
`train/gptsovits.py`. Built from the upstream repo's own installer:

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS third_party/GPT-SoVITS
cd third_party/GPT-SoVITS && bash install.sh --device CU128 --source HF
```

Then two fixes recon found necessary on the working box:

```bash
# 1) installer pulls torchaudio for CUDA 13 → force back to cu128
conda run -n gptsovits pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps torchaudio
# 2) NPP build matched to CUDA 12.8 (note 12.3.3.100 here, NOT 12.4.1.87)
conda run -n gptsovits pip install nvidia-npp-cu12==12.3.3.100
```

This env has **no** persistent NPP activate hook (unlike the other three) — the
`serve_engines.sh` launcher exports `LD_LIBRARY_PATH` for it instead. You can
copy the hook in to make it persistent:

```bash
cp ~/miniforge3/envs/xtts/etc/conda/activate.d/zz_nvidia_libs.sh \
   ~/miniforge3/envs/gptsovits/etc/conda/activate.d/zz_nvidia_libs.sh
```

### Verify the envs

```bash
# repeat per env (xtts / f5tts / chatterbox / gptsovits)
conda run -n xtts python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

Expect `2.11.0+cu128 12.8 True (12, 0)`.

---

## 3. Models & assets

All of these land in **gitignored** directories. Paths below are relative to the
repo root.

### XTTS-v2 base (needed for both XTTS voices)

```bash
huggingface-cli download coqui/XTTS-v2 --local-dir third_party/xtts_v2_model
```

The studio pre-flight requires `third_party/xtts_v2_model/{config.json,
model.pth, vocab.json, dvae.pth, mel_stats.pth}`. The XTTS fine-tune's
`config.json` also points its tokenizer at this base `vocab.json`, so **both**
XTTS voices need this dir.

### F5 German checkpoint (only for the German toggle)

```bash
huggingface-cli download aihpi/F5-TTS-German --local-dir third_party/f5_tts_german
```

Needs `F5TTS_Base/model_365000.safetensors` + `vocab.txt`. Only required for the
German (`de`) F5 zero-shot voice. (Pre-flight does **not** check this file, so
German can still fail at generate-time even when pre-flight is green.)

### GPT-SoVITS (NOT used by the studio notebook)

Covered in step 2 above (`git clone` + `install.sh`). Skip unless you are using
the older `serve_*.sh` / A/B / train paths.

### Reference audio — the bring-your-own-voice reality

The original reference clip `experiments/danil/reference/danill_ref9.wav` is the
consenting subject's **private** audio and is **not** redistributable. Every
studio voice conditions on it.

**Important caveat:** as of this writing, the committed studio/notebook has **no
upload widget** and `studio.REF_CLIP` is a hardcoded module constant — the
notebook cells expose only `TEXT` / `VOICE` / `LANG`, and the design spec lists
"upload-your-own-voice" as out of scope. So "bring your own voice" today means
**editing the code/config to point at your own clip**, not clicking an upload
button.

To clone **your own** voice on a fresh machine:

1. Record a clean ~6–12 s clip of your voice (mono WAV is safest) and save it
   somewhere in the repo, e.g. `experiments/<you>/reference/myref.wav`.
2. Point the studio at it. The simplest honest option is to edit the two module
   constants in `src/voxclone/clone/studio.py`:
   - `REF_CLIP` → the absolute (or repo-root-relative) path to your WAV.
   - `REF_TEXT` → the **verbatim** transcript of what you said in that clip
     (the F5 voices use it as prompt text; getting it right matters for F5
     quality).
   (If you prefer not to hardcode, you can wrap these in an env-var lookup such
   as `VOXCLONE_REF_CLIP` / `VOXCLONE_REF_TEXT` — but that requires a small code
   change, not just config.)

> If you only have the zero-shot voices (no `runs/` fine-tunes), you also want
> to relax the pre-flight cell so it does not hard-fail on the missing
> fine-tune checkpoints — see the note in step 5.

### Confirm exactly what is still missing

```bash
.venv/bin/python -c "from voxclone.clone import studio; print('missing envs:', studio.missing_envs()); print('missing assets:', studio.missing_assets())"
```

`missing_envs()` checks the `f5tts`/`xtts`/`chatterbox` env Python interpreters
exist. `missing_assets()` checks the reference clip, the F5/XTTS fine-tune
checkpoints, and the XTTS base files. Drive both lists to empty (or accept that
the voices behind a missing asset won't load).

---

## 4. Machine-specific paths (now repo-relative — usually nothing to edit)

The model/asset paths reached by the engine adapters and launch scripts are
**repo-root-relative** — derived from `Path(__file__)` (Python) or
`${BASH_SOURCE}` (scripts) — so the clone runs from **any** directory and you do
**not** need to edit paths per machine. `studio.py`, `synth/xtts.py`
(`MODEL_DIR`), `synth/f5.py` (German ckpt/vocab), `scripts/serve_*.sh`, and
`train/gptsovits.py` were all de-hardcoded; a
`tests/test_no_hardcoded_paths.py` guard keeps it that way.

Two machine **assumptions** remain (adjust only if they don't hold for you):

- **conda lives at `$HOME/miniforge3`** with envs named
  `xtts` / `f5tts` / `chatterbox` / `gptsovits`. The studio derives each engine's
  Python from `Path.home()/"miniforge3"/"envs"/<name>`. If your miniforge is
  elsewhere or your envs are named differently, edit the `CONDA_ENVS` mapping in
  `src/voxclone/clone/studio.py`; the shell launchers honor an override
  (`CONDA=/path/to/bin/conda bash scripts/serve_engines.sh`).
- **the reference clip** is still a hardcoded `studio.REF_CLIP` constant — see
  step 3 for bring-your-own-voice (this is the one path you may still need to set).

---

## 5. Run the studio notebook

1. Open the repo folder in **VS Code**. The committed `.vscode/settings.json`
   sets `"jupyter.notebookFileRoot": "${workspaceFolder}"`, which makes the
   notebook's working directory the repo root — required so `voxclone` imports
   and the studio's `ROOT` (and its asset list) resolve. If you run the notebook
   another way, ensure the cwd is the repo root.
2. Open `notebooks/voice_cloning_study.ipynb` and select the kernel **`VoxClone (.venv)`**
   (the one you registered in step 1).
3. **Run All.** The analysis cells (pipeline, results, the "≈ 90 %" derivation)
   run **GPU-free** and immediately. The **clone-your-own-voice** panel — and, in
   the author's private build, the **hear-my-voice** cell — synthesize live; the
   first clip per voice warms the GPU (~60–90 s).
   - **Pre-flight** reports `studio.missing_assets()` / `studio.missing_envs()`
     and sets `CAN_GENERATE`. If assets/envs are missing it does **not** abort —
     the audio cells just print a "run me on the GPU box" note and the rest of the
     notebook still runs. (Only a wrong kernel raises a friendly `SystemExit`.)

**How the servers start.** The studio is **zero-terminal**: on `studio.say(...)`
it auto-launches the right engine's FastAPI server as a background subprocess in
that engine's conda env (interpreter by absolute path), waits for `GET /health`,
then POSTs the text and plays back the result. You do **not** run any terminal
launcher. Logs go to `/tmp/vox_<key>_<port>.log`; PIDs to
`/tmp/vox_studio_<port>.pid`. Only **one** warm worker runs at a time — switching
voice tears down the previous worker to free VRAM (so the first clip after a
switch re-warms). Generated WAVs are saved under `notebooks/clone_outputs/`.

The studio uses its own ports to avoid a collision the shared launcher has:
`f5_finetuned` 9892, `f5_zeroshot` 9882, `xtts_finetuned` 9891, `xtts_zeroshot`
9881, `chatterbox` 9883.

> There is also a **manual** launcher, `scripts/serve_engines.sh` (conda-run
> based, ports 9880–9883), but the notebook does **not** use it — it is a
> separate path and not required for the studio. Use it only for the older A/B
> workflow.

**The interactive flow.** The last code cell builds an `ipywidgets` panel:

- **Language** dropdown: English / Deutsch. Switching to Deutsch lazily loads
  the German path per engine (F5 German checkpoint, XTTS multilingual base,
  Chatterbox multilingual). `f5_finetuned` is English-only and is dropped from
  the dropdown when Deutsch is selected; German defaults to `f5_zeroshot`.
- **Voice** dropdown: the five voices (subject to language).
- **Textarea**: type any sentence.
- **Generate** synthesizes and plays it; **Restart-voice** reloads the worker.
- **Bring your own voice:** the clone-your-own-voice section has a **file-upload**
  widget — upload a ~6–12 s mono WAV of yourself, paste its transcript, pick a
  zero-shot engine, and it clones *your* voice (no training, no terminal).

If `ipywidgets` fails to render, the panel degrades gracefully and you can keep
using the auto-demo cell (edit `TEXT` / `VOICE` / `LANG` and re-run).

**Headless smoke** (needs the kernel + GPU + all envs/assets):

```bash
.venv/bin/python -m nbconvert --to notebook --execute notebooks/voice_cloning_study.ipynb --ExecutePreprocessor.kernel_name=voxclone
```

---

## 6. Run the tests

The GPU-free suite (~323 tests) confirms the `.venv` is sane without any
engine env, model, or GPU:

```bash
uv run --extra dev pytest -q
```

---

## Troubleshooting

- **Wrong Python / `soccer` env gotcha.** The shell on the original box defaults
  to conda env `soccer`. Always invoke the project interpreter explicitly:
  `.venv/bin/python ...` (or `uv run --extra dev ...`). The notebook side of
  this is handled by selecting the `VoxClone (.venv)` kernel.

- **VS Code Jupyter kernel hangs on startup.** `ipykernel` must be `>=6,<7`
  (`pyproject.toml` already pins this). `ipykernel` 7.x hangs VS Code's Jupyter
  kernel startup.

- **`SystemExit` in the pre-flight cell.** Either you selected the wrong kernel
  (must be `VoxClone (.venv)`) or `missing_assets()`/`missing_envs()` is
  non-empty. Run the confirm command in step 3 and resolve each item. If you
  only have zero-shot voices, relax the pre-flight (step 5).

- **`libnppicc.so.12` not found / audio decode fails** (any engine). The NPP fix
  isn't on `LD_LIBRARY_PATH`. The studio injects it via `worker_env()`, so this
  is usually fine inside the notebook; if you launch a server manually, ensure
  `nvidia-npp-cu12` is installed in that env and the `zz_nvidia_libs.sh`
  activate hook exists (step 2). GPT-SoVITS specifically needs
  `LD_LIBRARY_PATH` set at launch (it has no persistent hook unless you copied
  one in).

- **`ModuleNotFoundError: voxclone` / asset paths not found / `ROOT` wrong.**
  The notebook's working directory isn't the repo root. In VS Code the committed
  `.vscode/settings.json` fixes this; otherwise set cwd to the repo root before
  launching.

- **"Server down" / health-check failure / voice won't load.** Check the
  worker's log at `/tmp/vox_<key>_<port>.log`. Common causes: the engine's conda
  env or a model asset is missing (re-run the step-3 confirm command); a stale
  worker holds the port (the studio tries to reclaim it only if
  `/proc/<pid>/cmdline` confirms it's ours, else fails loud — kill the stale
  process); or the engine hit a stale hardcoded absolute path (step 4).

- **First clip / first clip after switching voice is slow.** Expected: ~60–90 s
  GPU warm-up. Only one worker is warm at a time; switching voice tears the
  prior one down and re-warms.

- **Offline / first-run download failures.** The studio sets `HF_HUB_OFFLINE=1`
  and `TRANSFORMERS_OFFLINE=1`, assuming each engine env's HuggingFace cache is
  already warm. On a fresh machine the F5/Chatterbox/XTTS auto-downloads (and
  the Chatterbox multilingual + F5 German models) will **fail offline**. Fix:
  warm the caches once **online** (temporarily unset those two env vars for the
  first run, or run each engine once with networking enabled), then they read
  locally thereafter.

- **German fails even though pre-flight is green.** Pre-flight does not check the
  F5 German checkpoint or the Chatterbox multilingual model. Make sure
  `third_party/f5_tts_german/...` exists (step 3) and the German model caches are
  warm.

---

## See also

- `docs/HANDOFF.md` — live project state + the authoritative per-engine install
  runbooks (Workstream ② + GPT-SoVITS). Link target for env-build details.
- `docs/RUNBOOK_finetune.md` — produces the `runs/` checkpoints the two
  fine-tuned studio voices need (requires the private dataset).
- `docs/finetune-vs-zeroshot-report.md` — the fine-tune vs zero-shot results.
