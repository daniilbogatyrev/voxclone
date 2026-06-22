import numpy as np
from voxclone.common.audio import load_audio, save_audio, resample_audio

def test_save_then_load_roundtrip(tmp_path):
    sr = 48000
    audio = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    p = tmp_path / "tone.wav"
    save_audio(p, audio, sr)
    loaded, loaded_sr = load_audio(p)
    assert loaded_sr == sr
    assert loaded.dtype == np.float32
    assert loaded.ndim == 1
    assert np.allclose(loaded, audio, atol=1e-3)

def test_resample_changes_length_proportionally():
    sr = 48000
    audio = np.zeros(sr, dtype=np.float32)
    out = resample_audio(audio, sr, 24000)
    assert abs(len(out) - 24000) <= 2
