import numpy as np
from voxclone.prep.denoise import denoise_audio

def test_disabled_is_noop():
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    out = denoise_audio(audio, 48000, enabled=False)
    assert np.array_equal(out, audio)

def test_enabled_calls_denoiser():
    audio = np.ones(5, dtype=np.float32)
    called = {}
    def fake(a, sr):
        called["yes"] = True
        return a * 0.5
    out = denoise_audio(audio, 48000, enabled=True, denoiser=fake)
    assert called.get("yes") is True
    assert np.allclose(out, 0.5)
