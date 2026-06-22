import numpy as np
from pydantic import BaseModel

class Transcription(BaseModel):
    text: str
    confidence: float

def transcribe_clip(audio: np.ndarray, sr: int, model) -> Transcription:
    words = model.transcribe(audio, sr)
    words = [w for w in words if w["word"].strip()]
    if not words:
        return Transcription(text="", confidence=0.0)
    text = " ".join(w["word"].strip() for w in words).strip()
    confidence = float(np.mean([w["probability"] for w in words]))
    return Transcription(text=text, confidence=confidence)

class WhisperXModel:  # pragma: no cover (model)
    """Wraps faster-whisper + whisperx to emit word dicts with probabilities.

    `language` (ISO code, e.g. "en"/"de") forces the ASR language and selects the matching
    forced-aligner — leaving it on auto-detect mangles German (mis-detection + an English
    aligner over German audio yield bogus word scores)."""
    def __init__(self, model_name: str, device: str, language: str = "en"):
        import whisperx
        self._wx = whisperx
        self._device = device
        self._language = language
        self._asr = whisperx.load_model(model_name, device, compute_type="float16",
                                        language=language)
        self._align, self._meta = whisperx.load_align_model(
            language_code=language, device=device
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> list[dict]:
        result = self._asr.transcribe(audio, batch_size=16, language=self._language)
        aligned = self._wx.align(result["segments"], self._align, self._meta,
                                 audio, self._device, return_char_alignments=False)
        words = []
        for seg in aligned["segments"]:
            for w in seg.get("words", []):
                words.append({"word": w["word"],
                              "probability": float(w.get("score", 0.0))})
        return words

def load_transcriber(model_name: str, device: str, language: str = "en") -> "WhisperXModel":  # pragma: no cover
    return WhisperXModel(model_name, device, language)
