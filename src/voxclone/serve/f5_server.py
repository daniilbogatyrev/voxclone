"""Thin per-engine F5-TTS inference server.

Runs INSIDE the `f5tts` conda env (so the heavy `f5_tts` import never enters the
project kernel) and listens on a distinct port (F5 = 9882; XTTS = 9881; GPT-SoVITS
= 9880). Two endpoints mirror how the synth adapter drives it:

  POST /load_checkpoint {checkpoint_dir, vocab_file}  -> bind the finetuned safetensors
  POST /tts {text, reference_clip, prompt_text, ...}  -> 24 kHz float32 mono WAV bytes

F5 conditions on BOTH the reference clip and its transcript (`prompt_text`), so /tts
requires the enrollment clip plus its verbatim transcript. The real f5_tts.api.F5TTS
load + infer is behind a `# pragma: no cover` seam; the test injects a fake model via
`make_app(model_factory=...)` so the app is exercised GPU-free.
"""

import glob
import io
import os

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from pydantic import BaseModel

SAMPLE_RATE = 24000


def _resolve_checkpoint_file(checkpoint_path: str) -> str:
    """Resolve a concrete checkpoint FILE from a checkpoint dir (or pass a file through).

    ``F5TTS(ckpt_file=...)`` requires a FILE: ``load_checkpoint`` does
    ``ckpt_path.split(".")[-1]`` then ``load_file()``/``torch.load()`` — handing it a
    directory raises. The F5 trainer copies ``model_last.pt`` into the run dir, so:

      * if ``checkpoint_path`` is already a file, use it as-is;
      * else prefer ``<dir>/model_last.pt`` when present;
      * else pick the newest ``model_*.pt`` / ``*.safetensors`` in the dir;
      * else fall back to the original path (let F5TTS surface the error).
    """
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    if os.path.isdir(checkpoint_path):
        last = os.path.join(checkpoint_path, "model_last.pt")
        if os.path.isfile(last):
            return last
        candidates = (
            glob.glob(os.path.join(checkpoint_path, "model_*.pt"))
            + glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
        )
        if candidates:
            return max(candidates, key=os.path.getmtime)
    return checkpoint_path


def _resolve_vocab_file(checkpoint_path: str, vocab_file: str | None) -> str | None:
    """Pick the vocab file: explicit wins; else ``<dir>/vocab.txt`` if it exists; else None.

    The F5 trainer copies ``vocab.txt`` next to ``model_last.pt``, so a bare run dir
    still yields the matching tokenizer vocab without the caller spelling it out.
    """
    if vocab_file:
        return vocab_file
    if os.path.isdir(checkpoint_path):
        vocab = os.path.join(checkpoint_path, "vocab.txt")
        if os.path.isfile(vocab):
            return vocab
    return None


class LoadReq(BaseModel):
    checkpoint_dir: str
    vocab_file: str | None = None


class TTSReq(BaseModel):
    text: str
    reference_clip: str
    prompt_text: str = ""
    nfe_step: int = 32
    cfg_strength: float = 2.0
    speed: float = 1.0
    seed: int = 42


def _default_f5_factory(model, ckpt_file, vocab_file):  # pragma: no cover (model)
    """Construct the real ``f5_tts.api.F5TTS`` (imported lazily, runs only in-env)."""
    from f5_tts.api import F5TTS

    return F5TTS(model=model, ckpt_file=ckpt_file, vocab_file=vocab_file)


def _default_model_factory(f5_factory=None):
    """Build the F5-TTS model wrapper.

    Returns an object exposing `.load_checkpoint(checkpoint_dir, vocab_file)`,
    `.tts(text, reference_clip, prompt_text, nfe_step, cfg_strength, speed, seed)`,
    and `.sr == 24000`. ``f5_factory`` is the seam that constructs the real
    `f5_tts.api.F5TTS` (kept behind a `# pragma: no cover` default); tests inject a
    fake to exercise the ckpt-dir -> FILE resolution GPU-free.
    """
    f5_factory = f5_factory or _default_f5_factory

    class _F5Model:
        sr = SAMPLE_RATE

        def __init__(self):
            self._model = None
            self._ckpt = None
            self._vocab = None

        def load_checkpoint(self, checkpoint_dir, vocab_file=None):
            # F5TTS needs a concrete FILE, not the run dir; the trainer copies
            # model_last.pt + vocab.txt into the dir so this resolution finds them.
            ckpt_file = _resolve_checkpoint_file(checkpoint_dir)
            vocab = _resolve_vocab_file(checkpoint_dir, vocab_file)
            self._ckpt = ckpt_file
            self._vocab = vocab
            self._model = f5_factory(
                model="F5TTS_v1_Base",
                ckpt_file=ckpt_file,
                vocab_file=vocab,
            )

        def tts(self, text, reference_clip, prompt_text="", nfe_step=32,
                cfg_strength=2.0, speed=1.0, seed=42):
            wav, sr, _ = self._model.infer(
                ref_file=reference_clip,
                ref_text=prompt_text,
                gen_text=text,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                speed=speed,
                seed=seed,
            )
            return np.asarray(wav, dtype=np.float32), int(sr)

    return _F5Model()


def make_app(model_factory=None) -> FastAPI:
    app = FastAPI(title="f5-server")
    state = {"model": None, "factory": model_factory or _default_model_factory}

    def _model():
        if state["model"] is None:
            state["model"] = state["factory"]()
        return state["model"]

    @app.get("/health")
    def health() -> dict:
        return {"server": "f5", "ready": True}

    @app.post("/load_checkpoint")
    def load_checkpoint(req: LoadReq) -> dict:
        _model().load_checkpoint(req.checkpoint_dir, req.vocab_file)
        return {"loaded": req.checkpoint_dir}

    @app.post("/tts")
    def tts(req: TTSReq) -> Response:
        audio, sr = _model().tts(
            req.text, req.reference_clip, prompt_text=req.prompt_text,
            nfe_step=req.nfe_step, cfg_strength=req.cfg_strength,
            speed=req.speed, seed=req.seed,
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9882)
    args = ap.parse_args()
    uvicorn.run(make_app(), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
