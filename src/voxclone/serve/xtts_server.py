"""Thin per-engine XTTS-v2 inference server.

Persistent FastAPI server run INSIDE the ``xtts`` conda env (mirrors how
``GPTSoVITSSynth`` talks to api_v2) on a distinct port (XTTS 9881; GPT-SoVITS
9880, F5 9882). Two endpoints:

  * ``POST /load_checkpoint {checkpoint_dir}`` -> binds the active checkpoint
    SERVER-SIDE (eval calls ``synthesize(text, ref, {})`` with EMPTY params, so
    the checkpoint must be bound here, not passed through ``/tts``).
  * ``POST /tts`` -> WAV float32 bytes (``audio/wav``), downmixed mono, 24 kHz.

The real model build + the XTTS pinned gotchas (``transformers<=4.57.6`` and
``torch.serialization.add_safe_globals([...])`` before any checkpoint load) live
in ``_default_model_factory`` (``# pragma: no cover`` — only runs in-env). Tests
inject a fake ``model_factory`` so ``make_app`` is exercised GPU-free.

Conditioning latents are cached keyed by ``reference_clip`` (XTTS recomputes
``get_conditioning_latents`` per voice otherwise — expensive); the cache logic
is exercised by the tests, the real latent computation is in the model.
"""

import glob
import io
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from pydantic import BaseModel

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9881
TARGET_SR = 24000


def _resolve_xtts_checkpoint(checkpoint_dir: str) -> str:
    """Resolve the concrete XTTS weights FILE from a Coqui run dir.

    ``Xtts.load_checkpoint`` defaults ``checkpoint_dir`` to ``<dir>/model.pth``,
    but the Coqui trainer writes ``best_model.pth`` / ``best_model_<step>.pth`` /
    ``checkpoint_<step>.pth`` — no bare ``model.pth``. Resolution order, choosing
    the BEST-eval model over the last training step and never an auxiliary weight
    (``dvae.pth`` / ``mel_stats.pth`` / ``speakers_xtts.pth``):

      1. an explicit file path (passed through);
      2. ``best_model.pth`` then ``model.pth`` by exact name;
      3. ``best_model_<step>.pth`` with the highest step;
      4. ``checkpoint_<step>.pth`` with the highest step;
      5. fall back to ``<dir>/model.pth`` (let Xtts surface the error).
    """
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir
    for name in ("best_model.pth", "model.pth"):
        cand = os.path.join(checkpoint_dir, name)
        if os.path.isfile(cand):
            return cand

    def _step(path: str) -> int:
        m = re.search(r"_(\d+)\.pth$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    for pattern in ("best_model_*.pth", "checkpoint_*.pth"):
        cands = glob.glob(os.path.join(checkpoint_dir, pattern))
        if cands:
            return max(cands, key=lambda p: (_step(p), os.path.getmtime(p)))
    return os.path.join(checkpoint_dir, "model.pth")


class LoadReq(BaseModel):
    checkpoint_dir: str


class TTSReq(BaseModel):
    text: str
    reference_clip: str
    language: str = "en"
    temperature: float = 0.7


def _default_model_factory():  # pragma: no cover (model — runs only in the xtts conda env)
    import torch
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig

    # XTTS gotcha #2: register safe globals before any checkpoint load (PyTorch
    # 2.6 weights_only default). Gotcha #1 (transformers<=4.57.6) is pinned in
    # the xtts env's install, not here.
    torch.serialization.add_safe_globals(
        [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]
    )

    class _XTTSModel:
        sr = TARGET_SR

        def __init__(self):
            self.config = None
            self.model = None
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        def load_checkpoint(self, checkpoint_dir):
            config = XttsConfig()
            config.load_json(f"{checkpoint_dir}/config.json")
            # The Coqui trainer bakes ABSOLUTE base-model paths into a fine-tune's
            # config.json (tokenizer / mel-norm / dvae / base weights). A fine-tune
            # trained on another machine therefore points at paths that don't exist
            # here -> "vocab.json file not found". Re-point all four at THIS repo's
            # base XTTS model dir (repo-root-relative) before building the model.
            base = Path(__file__).resolve().parents[3] / "third_party" / "xtts_v2_model"
            ma = config.model_args
            ma.tokenizer_file = str(base / "vocab.json")
            ma.mel_norm_file = str(base / "mel_stats.pth")
            ma.dvae_checkpoint = str(base / "dvae.pth")
            ma.xtts_checkpoint = str(base / "model.pth")
            model = Xtts.init_from_config(config)
            # Coqui defaults checkpoint_dir -> <dir>/model.pth; the trainer wrote
            # best_model.pth. Pass the resolved FILE via checkpoint_path.
            ckpt_path = _resolve_xtts_checkpoint(checkpoint_dir)
            model.load_checkpoint(config, checkpoint_path=ckpt_path, use_deepspeed=False)
            model.to(self.device)
            self.config, self.model = config, model

        def get_conditioning_latents(self, reference_clip):
            return self.model.get_conditioning_latents(audio_path=[reference_clip])

        def tts(self, text, reference_clip, latents=None, language="en", temperature=0.7):
            gpt_cond_latent, speaker_embedding = latents
            out = self.model.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=temperature,
            )
            return np.asarray(out["wav"], dtype=np.float32), self.sr

    return _XTTSModel()


def make_app(model_factory=None) -> FastAPI:
    """Build the XTTS inference app.

    ``model_factory`` is dependency-injected (tests pass a GPU-free fake); the
    real model is built lazily on first request via ``_default_model_factory``.
    """
    app = FastAPI(title="xtts-server")
    state = {"model": None, "latents": {}, "factory": model_factory or _default_model_factory}

    def _model():
        if state["model"] is None:
            state["model"] = state["factory"]()
        return state["model"]

    @app.get("/health")
    def health() -> dict:
        return {"server": "xtts", "ready": True}

    def _conditioning(model, reference_clip):
        # Cache keyed by reference_clip (the only conditioning input that
        # changes between requests for a given checkpoint).
        cache = state["latents"]
        if reference_clip not in cache:
            cache[reference_clip] = model.get_conditioning_latents(reference_clip)
        return cache[reference_clip]

    @app.post("/load_checkpoint")
    def load_checkpoint(req: LoadReq) -> dict:
        model = _model()
        model.load_checkpoint(req.checkpoint_dir)
        # New checkpoint invalidates any cached latents (they are checkpoint-bound).
        state["latents"] = {}
        return {"loaded": req.checkpoint_dir}

    @app.post("/tts")
    def tts(req: TTSReq) -> Response:
        model = _model()
        latents = _conditioning(model, req.reference_clip)
        audio, sr = model.tts(
            req.text,
            req.reference_clip,
            latents=latents,
            language=req.language,
            temperature=req.temperature,
        )
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, int(sr), format="WAV", subtype="FLOAT")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    return app


def main() -> None:  # pragma: no cover (server entry)
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="XTTS-v2 thin inference server")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    uvicorn.run(make_app(), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
