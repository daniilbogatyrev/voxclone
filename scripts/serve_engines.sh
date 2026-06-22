#!/usr/bin/env bash
# Launch all four engine servers, each in its own conda env. Run from the repo root.
# Usage: bash scripts/serve_engines.sh   (Ctrl-C stops the foreground one; others are backgrounded)
# Each server is started with `conda run -n <env> ...` against the per-engine env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-$HOME/miniforge3/bin/conda}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# GPT-SoVITS: its own api_v2 server (needs the NPP/LD_LIBRARY_PATH fix), port 9880
"$CONDA" run --no-capture-output -n gptsovits bash -c '
  cd '"$ROOT"'/third_party/GPT-SoVITS
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/npp/lib:$LD_LIBRARY_PATH"
  export TERM=xterm
  python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml' &

# XTTS / F5 / Chatterbox: the uniform engine_server in each env
"$CONDA" run --no-capture-output -n xtts bash -c \
  'cd '"$ROOT"' && PYTHONPATH='"$ROOT"'/src COQUI_TOS_AGREED=1 python -m voxclone.serve.engine_server --engine xtts --port 9881' &
"$CONDA" run --no-capture-output -n f5tts bash -c \
  'cd '"$ROOT"' && PYTHONPATH='"$ROOT"'/src python -m voxclone.serve.engine_server --engine f5 --port 9882' &
"$CONDA" run --no-capture-output -n chatterbox bash -c \
  'cd '"$ROOT"' && PYTHONPATH='"$ROOT"'/src python -m voxclone.serve.engine_server --engine chatterbox --port 9883' &

wait
