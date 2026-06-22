from pathlib import Path
from dataclasses import dataclass
import numpy as np
from voxclone.common.audio import clipped_fraction, save_audio

CALIBRATION_SENTENCE = (
    "This is my natural speaking voice, calm, clear, and steady, "
    "as I read these few lines aloud today."
)

def clip_path(raw_dir: str | Path, category: str, index: int) -> Path:
    return Path(raw_dir) / f"{category}_{index:04d}.wav"

def clip_stats(audio: np.ndarray, sr: int) -> dict:
    return {
        "duration_s": len(audio) / sr if sr else 0.0,
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "clipped_fraction": clipped_fraction(audio),
    }

@dataclass
class SessionTotals:
    clip_count: int = 0
    total_seconds: float = 0.0

    def add(self, duration_s: float) -> None:
        self.clip_count += 1
        self.total_seconds += duration_s

    @property
    def total_minutes(self) -> float:
        return self.total_seconds / 60

def session_cap_reached(totals: "SessionTotals", max_minutes: float = 20.0) -> bool:
    """True once a recording session has captured >= max_minutes of speech
    (guards against fatigue drift across a long session)."""
    return totals.total_minutes >= max_minutes

def _record_blocking(seconds: float, sr: int) -> np.ndarray:  # pragma: no cover (hardware)
    import sounddevice as sd
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return rec.reshape(-1)

def record_clip(raw_dir: str | Path, category: str, index: int, seconds: float,
                sr: int, recorder=_record_blocking) -> tuple[Path, dict]:
    audio = recorder(seconds, sr)
    path = clip_path(raw_dir, category, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(path, audio, sr)
    return path, clip_stats(audio, sr)
