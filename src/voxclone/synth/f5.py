"""F5-TTS adapters.

Two complementary adapters with DIFFERENT call signatures for DIFFERENT consumers:

* ``F5Adapter`` (P02-T1, notebook-facing): ``synthesize(text, ref_path, ref_text="",
  params=None)``, in-process lazy ``f5_tts`` import.
* ``F5Synth`` (P07-T7, eval/serve-facing SynthAdapter): checkpoint-bound at construction,
  ``synthesize(text, reference_clip, params) -> (np.float32 mono, int sr)``. Mirrors
  ``GPTSoVITSSynth`` and posts to the per-engine ``f5_server`` (``/load_checkpoint`` from
  ``self.checkpoint`` THEN ``/tts``). The checkpoint MUST be bound at construction because
  ``eval/runner.py`` calls ``synthesize(text, reference_clip, {})`` with EMPTY params -- it
  must NOT regress to the GPT-SoVITS bug of reading finetuned weights out of ``params``.
  F5 also conditions on the reference clip AND its transcript (``params['prompt_text']``).
"""
import io
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

# Per-engine inference server (runs inside the `f5tts` conda env). F5 9882,
# XTTS 9881, GPT-SoVITS 9880.
DEFAULT_SERVER_URL = "http://127.0.0.1:9882"

# F5's base model is English+Chinese only; German needs a German-trained checkpoint.
# aihpi/F5-TTS-German, vocos variant (matches F5's default vocoder; no bigvgan setup), with
# its OWN vocab.txt. Loaded lazily on the first `language="de"` request (see F5Adapter).
GERMAN_MODEL = "F5TTS_Base"  # the aihpi checkpoint is the F5TTS_Base (v0) architecture
# Repo-root-relative (src/voxclone/synth/ -> repo root) so a clone runs from any path.
_GERMAN_DIR = Path(__file__).resolve().parents[3] / "third_party" / "f5_tts_german"
GERMAN_CKPT = str(_GERMAN_DIR / "F5TTS_Base" / "model_365000.safetensors")
GERMAN_VOCAB = str(_GERMAN_DIR / "vocab.txt")


def _real_generate_f5(text, reference_clip, checkpoint, params):
    """Synthesize via the per-engine f5_server. The checkpoint is bound at construction,
    so we ``/load_checkpoint`` it here from ``checkpoint`` (NOT params) before ``/tts``.
    F5 conditions on the reference clip AND its transcript (``params['prompt_text']``).
    Returns (float32 mono, 24000).
    """
    base = params.get("server_url", DEFAULT_SERVER_URL)
    timeout = params.get("timeout", 120.0)
    httpx.post(
        f"{base}/load_checkpoint",
        json={"checkpoint_dir": checkpoint},
        timeout=timeout,
    ).raise_for_status()
    payload = {
        "text": text,
        "reference_clip": reference_clip,
        "prompt_text": params.get("prompt_text", ""),
        "nfe_step": params.get("nfe_step", 32),
        "cfg_strength": params.get("cfg_strength", 2.0),
        "speed": params.get("speed", 1.0),
        "seed": params.get("seed", 42),
    }
    resp = httpx.post(f"{base}/tts", json=payload, timeout=timeout)
    resp.raise_for_status()
    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio.astype(np.float32), int(sr)


class F5Synth:
    """Eval/serve-facing checkpoint-bound F5 SynthAdapter (mirrors GPTSoVITSSynth)."""

    def __init__(self, checkpoint: str, generate_fn=_real_generate_f5):
        self.checkpoint = checkpoint
        self.generate_fn = generate_fn

    def synthesize(self, text: str, reference_clip: str, params: dict) -> tuple[np.ndarray, int]:
        audio, sr = self.generate_fn(text, reference_clip, self.checkpoint, params)
        return np.asarray(audio, dtype=np.float32), int(sr)


class F5Adapter:
    def __init__(self, model: str = "F5TTS_v1_Base", device: str = "cuda", _synth=None):
        self.model_name, self.device, self._synth, self._f5 = model, device, _synth, None
        self._f5_de = None  # German checkpoint, lazily loaded on first de request

    def load(self) -> None:
        if self._synth is not None:
            return
        from f5_tts.api import F5TTS  # pragma: no cover
        self._f5 = F5TTS(model=self.model_name, device=self.device)

    def _german(self):  # pragma: no cover (gpu)
        """Lazily load the German checkpoint (its own arch + vocab) and cache it."""
        if self._f5_de is None:
            from f5_tts.api import F5TTS
            self._f5_de = F5TTS(model=GERMAN_MODEL, ckpt_file=GERMAN_CKPT,
                                vocab_file=GERMAN_VOCAB, device=self.device)
        return self._f5_de

    def synthesize(self, text, ref_path, ref_text="", params=None):
        params = params or {}
        if self._synth is not None:
            return self._synth(text, ref_path, ref_text, params)
        f5 = self._german() if params.get("language") == "de" else self._f5  # pragma: no cover
        wav, sr, _ = f5.infer(ref_file=ref_path, ref_text=ref_text, gen_text=text,  # pragma: no cover
                              nfe_step=params.get("nfe_step", 32),
                              cfg_strength=params.get("cfg_strength", 2.0),
                              speed=params.get("speed", 1.0), seed=params.get("seed", 42))
        return np.asarray(wav, dtype="float32"), int(sr)
