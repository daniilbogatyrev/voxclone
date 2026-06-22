"""XTTS-v2 adapters.

Two complementary adapters live here, with DIFFERENT call signatures for DIFFERENT
consumers (do not conflate them):

* ``XTTSAdapter`` (P02-T1, notebook-facing): ``synthesize(text, ref_path, ref_text="",
  params=None)``, in-process lazy ``TTS`` import, driven by the uniform notebook client.
* ``XTTSSynth`` (P07-T7, eval/serve-facing SynthAdapter): checkpoint-bound at construction,
  ``synthesize(text, reference_clip, params) -> (np.float32 mono, int sr)``. It mirrors
  ``GPTSoVITSSynth`` and posts to the per-engine ``xtts_server`` (``/load_checkpoint`` from
  ``self.checkpoint`` THEN ``/tts``). The checkpoint MUST be bound at construction because
  ``eval/runner.py`` calls ``synthesize(text, reference_clip, {})`` with EMPTY params -- it
  must NOT regress to the latent GPT-SoVITS bug of reading the finetuned weights out of
  ``params``.

The heavy ``TTS`` import stays lazy / out-of-process, so importing this module is GPU-free.
"""
import io
import os
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

# Repo-root-relative so a clone runs from any path (src/voxclone/synth/ -> repo root).
MODEL_DIR = str(Path(__file__).resolve().parents[3] / "third_party" / "xtts_v2_model")

# Per-engine inference server (runs inside the `xtts` conda env). XTTS 9881,
# GPT-SoVITS 9880, F5 9882.
DEFAULT_SERVER_URL = "http://127.0.0.1:9881"


def _real_generate_xtts(text, reference_clip, checkpoint, params):
    """Synthesize via the per-engine xtts_server (runs in the xtts conda env).

    The checkpoint is bound at construction, so we ``/load_checkpoint`` it here from
    ``checkpoint`` (NOT from ``params``) before ``/tts`` -- because eval calls
    ``synthesize(text, ref, {})`` with empty params. Returns (float32 mono, 24000).
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
        "language": params.get("language", "en"),
        "temperature": params.get("temperature", 0.7),
    }
    resp = httpx.post(f"{base}/tts", json=payload, timeout=timeout)
    resp.raise_for_status()
    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio.astype(np.float32), int(sr)


class XTTSSynth:
    """Eval/serve-facing checkpoint-bound XTTS SynthAdapter (mirrors GPTSoVITSSynth)."""

    def __init__(self, checkpoint: str, generate_fn=_real_generate_xtts):
        self.checkpoint = checkpoint
        self.generate_fn = generate_fn

    def synthesize(self, text: str, reference_clip: str, params: dict) -> tuple[np.ndarray, int]:
        audio, sr = self.generate_fn(text, reference_clip, self.checkpoint, params)
        return np.asarray(audio, dtype=np.float32), int(sr)


class XTTSAdapter:
    def __init__(self, model_dir: str = MODEL_DIR, device: str = "cuda", _synth=None):
        self.model_dir, self.device, self._synth = model_dir, device, _synth
        self._model, self._sr = None, 24000

    def load(self) -> None:
        if self._synth is not None:
            return
        import torch  # pragma: no cover
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
        from TTS.config.shared_configs import BaseDatasetConfig
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
        cfg = XttsConfig()
        cfg.load_json(os.path.join(self.model_dir, "config.json"))
        self._model = Xtts.init_from_config(cfg)
        self._model.load_checkpoint(cfg, checkpoint_dir=self.model_dir, use_deepspeed=False)
        self._model.to(self.device)
        self._sr = getattr(self._model.config.audio, "output_sample_rate", 24000)

    def synthesize(self, text, ref_path, ref_text="", params=None):
        params = params or {}
        if self._synth is not None:
            return self._synth(text, ref_path, ref_text, params)
        gpt_cond, spk = self._model.get_conditioning_latents(audio_path=[ref_path])  # pragma: no cover
        out = self._model.inference(text, language=params.get("language", "en"),
                                    gpt_cond_latent=gpt_cond, speaker_embedding=spk,
                                    temperature=params.get("temperature", 0.7))
        return np.asarray(out["wav"], dtype="float32"), int(self._sr)
