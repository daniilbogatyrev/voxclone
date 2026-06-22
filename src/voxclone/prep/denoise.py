import numpy as np

def _default_denoiser(audio: np.ndarray, sr: int) -> np.ndarray:  # pragma: no cover
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=sr).astype(np.float32)

def denoise_audio(audio: np.ndarray, sr: int, enabled: bool = True,
                  denoiser=_default_denoiser) -> np.ndarray:
    if not enabled:
        return audio
    return denoiser(audio, sr).astype(np.float32)
