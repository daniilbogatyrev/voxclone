"""Chatterbox adapters.

Two complementary adapters with DIFFERENT call signatures for DIFFERENT consumers:

* ``ChatterboxAdapter`` (P02-T1, notebook-facing): ``synthesize(text, ref_path, ref_text="",
  params=None)``, in-process lazy ``chatterbox`` import.
* ``ChatterboxSynth`` (eval/serve-facing SynthAdapter): checkpoint-bound at construction,
  ``synthesize(text, reference_clip, params) -> (np.float32 mono, int sr)``. Mirrors
  ``GPTSoVITSSynth`` so it slots into ``SYNTH_ENGINES`` / eval / serve uniformly. Chatterbox is
  ZERO-SHOT only (``TRAIN_ENGINES`` omits it -- it is never fine-tuned), so ``checkpoint`` is
  accepted purely for the uniform naming contract and never names finetuned weights; only the
  ``reference_clip`` conditions generation. The heavy ``chatterbox`` import stays inside the
  real generate path, so importing this module is GPU-free.
"""
import numpy as np


def _real_generate_chatterbox(text, reference_clip, checkpoint, params):  # pragma: no cover (gpu)
    """Zero-shot Chatterbox synthesis in-process (no per-engine server, no finetuned ckpt).

    ``checkpoint`` is ignored for weights (Chatterbox is zero-shot); only ``reference_clip``
    conditions the clone. Returns (float32 mono, model sample rate).
    """
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device=params.get("device", "cuda"))
    wav = model.generate(text, audio_prompt_path=reference_clip,
                         exaggeration=params.get("exaggeration", 0.5),
                         cfg_weight=params.get("cfg_weight", 0.5),
                         temperature=params.get("temperature", 0.8))
    audio = wav.squeeze().detach().cpu().numpy().astype("float32")
    return audio, int(model.sr)


class ChatterboxSynth:
    """Eval/serve-facing checkpoint-bound Chatterbox SynthAdapter (mirrors GPTSoVITSSynth).

    Chatterbox is zero-shot only: ``checkpoint`` is accepted for interface/naming uniformity
    but is never a fine-tuned weight (it is omitted from ``TRAIN_ENGINES``).
    """

    def __init__(self, checkpoint: str, generate_fn=_real_generate_chatterbox):
        self.checkpoint = checkpoint
        self.generate_fn = generate_fn

    def synthesize(self, text: str, reference_clip: str, params: dict) -> tuple[np.ndarray, int]:
        audio, sr = self.generate_fn(text, reference_clip, self.checkpoint, params)
        return np.asarray(audio, dtype=np.float32), int(sr)


class ChatterboxAdapter:
    def __init__(self, device: str = "cuda", _synth=None):
        self.device, self._synth, self._model = device, _synth, None
        self._ml = None  # multilingual model, lazily loaded for non-English languages

    def load(self) -> None:
        if self._synth is not None:
            return
        from chatterbox.tts import ChatterboxTTS  # pragma: no cover
        self._model = ChatterboxTTS.from_pretrained(device=self.device)

    def _multilingual(self):  # pragma: no cover (gpu)
        """The English ChatterboxTTS has no language control; non-English needs the separate
        ChatterboxMultilingualTTS. Loaded lazily so the English path/startup is unchanged."""
        if self._ml is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self._ml = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        return self._ml

    def synthesize(self, text, ref_path, ref_text="", params=None):
        params = params or {}
        if self._synth is not None:
            return self._synth(text, ref_path, ref_text, params)
        lang = params.get("language", "en")
        if lang and lang != "en":  # pragma: no cover (gpu)
            model = self._multilingual()
            # cfg_weight=0 mitigates the English accent bleeding into cross-lingual output.
            wav = model.generate(text, language_id=lang, audio_prompt_path=ref_path,
                                 exaggeration=params.get("exaggeration", 0.5),
                                 cfg_weight=params.get("cfg_weight", 0.0),
                                 temperature=params.get("temperature", 0.8))
            return wav.squeeze().detach().cpu().numpy().astype("float32"), int(model.sr)
        wav = self._model.generate(text, audio_prompt_path=ref_path,  # pragma: no cover
                                   exaggeration=params.get("exaggeration", 0.5),
                                   cfg_weight=params.get("cfg_weight", 0.5),
                                   temperature=params.get("temperature", 0.8))
        audio = wav.squeeze().detach().cpu().numpy().astype("float32")
        return audio, int(self._model.sr)
