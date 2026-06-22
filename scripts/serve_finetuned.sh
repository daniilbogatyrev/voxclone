#!/usr/bin/env bash
# Launch the two checkpoint-bound finetuned inference servers, each in its own
# conda env. Run from the repo root.
# Usage: bash scripts/serve_finetuned.sh   (Ctrl-C stops the foreground one; others are backgrounded)
#
# This is the LIVE eval/serve path (distinct from scripts/serve_engines.sh, which
# starts the notebook engine_server.py exposing /synth+/health). The eval/serve
# checkpoint-bound adapters (synth/xtts.py XTTSSynth, synth/f5.py F5Synth) POST to
# /load_checkpoint + /tts, which are served by:
#   * voxclone.serve.xtts_server  (port 9881, env `xtts`)
#   * voxclone.serve.f5_server    (port 9882, env `f5tts`)
# Each server is started with `conda run -n <env> ...` against the per-engine env,
# with the NVIDIA NPP lib dir prepended to LD_LIBRARY_PATH (the torchcodec
# libnppicc.so.12 fix from HANDOFF).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-$HOME/miniforge3/bin/conda}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# XTTS-v2 finetuned checkpoint server (env `xtts`), port 9881.
# COQUI_TOS_AGREED=1 (CPML); PYTHONPATH=src so `voxclone` imports; NPP fix on LD_LIBRARY_PATH.
"$CONDA" run --no-capture-output -n xtts bash -c '
  cd '"$ROOT"'
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH='"$ROOT"'/src
  export COQUI_TOS_AGREED=1
  python -m voxclone.serve.xtts_server --port 9881' &

# F5-TTS finetuned checkpoint server (env `f5tts`), port 9882.
# PYTHONPATH=src so `voxclone` imports; NPP fix on LD_LIBRARY_PATH.
"$CONDA" run --no-capture-output -n f5tts bash -c '
  cd '"$ROOT"'
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH='"$ROOT"'/src
  python -m voxclone.serve.f5_server --port 9882' &

wait
