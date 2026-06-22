from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, sr

def save_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_24")

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    return librosa.resample(
        audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
    ).astype(np.float32)

def clipped_fraction(audio: np.ndarray, threshold: float = 0.99) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))

def peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0.0:
        return audio.astype(np.float32)
    target_amp = 10 ** (target_dbfs / 20)
    return (audio * (target_amp / peak)).astype(np.float32)

def measure_lufs(audio: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(audio.astype(np.float64)))

def normalize_lufs(audio: np.ndarray, sr: int, target_lufs: float = -16.0) -> np.ndarray:
    current = measure_lufs(audio, sr)
    if not np.isfinite(current):
        return audio.astype(np.float32)
    out = pyln.normalize.loudness(audio.astype(np.float64), current, target_lufs)
    return out.astype(np.float32)

def estimate_snr_db(audio: np.ndarray, sr: int, frame_ms: float = 20.0) -> float:
    """Cheap single-clip SNR proxy: ratio of high- to low-energy frames (dB).

    Relies on speech having quiet gaps; a sustained tone reads as low SNR.
    Returns 0.0 for empty/too-short input."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size == 0:
        return 0.0
    frame = max(1, int(sr * frame_ms / 1000.0))
    n = audio.size // frame
    if n < 2:
        return 60.0
    energies = (audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    energies = energies[energies > 0]
    if energies.size < 2:
        return 0.0
    noise = float(np.percentile(energies, 10))
    signal = float(np.percentile(energies, 90))
    if noise <= 0 or signal <= 0:
        return 0.0
    return float(10.0 * np.log10(signal / noise))
