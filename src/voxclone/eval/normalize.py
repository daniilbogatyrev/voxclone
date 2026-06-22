import numpy as np
from voxclone.common.audio import normalize_lufs, peak_normalize, resample_audio

def normalize_for_eval(audio: np.ndarray, sr: int, target_lufs: float = -23.0,
                       target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """One front end every clip passes through before any metric, so the comparison
    is about the models, not their sample rate or loudness:
    mono -> loudness-normalize to target_lufs -> resample to target_sr -> peak-limit to <= -1 dBFS."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    audio = normalize_lufs(audio, sr, target_lufs)
    audio = resample_audio(audio, sr, target_sr)
    # peak-limit AFTER resampling: the resampling filter can overshoot, so limiting
    # beforehand would not actually bound the output.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 10 ** (-1.0 / 20):  # -1 dBFS
        audio = peak_normalize(audio, -1.0)
    return audio.astype(np.float32), target_sr
