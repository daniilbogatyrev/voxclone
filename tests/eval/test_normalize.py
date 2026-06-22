import numpy as np
from voxclone.eval.normalize import normalize_for_eval

def test_outputs_16k_mono_peak_limited():
    sr = 48000
    audio = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    out, osr = normalize_for_eval(audio, sr)
    assert osr == 16000
    assert out.ndim == 1
    assert out.shape[0] == 16000          # 1 s resampled 48k -> 16k
    assert float(np.max(np.abs(out))) <= 0.95   # limited near -1 dBFS
    assert np.isfinite(out).all()

def test_downmixes_stereo():
    sr = 16000
    stereo = np.zeros((sr, 2), dtype=np.float32)
    stereo[:, 0] = 0.2
    out, osr = normalize_for_eval(stereo, sr)
    assert out.ndim == 1 and osr == 16000

def test_silence_does_not_crash():
    out, osr = normalize_for_eval(np.zeros(48000, dtype=np.float32), 48000)
    assert osr == 16000 and out.shape[0] == 16000 and np.isfinite(out).all()

def test_custom_target_sr():
    out, osr = normalize_for_eval(np.zeros(48000, dtype=np.float32), 48000, target_sr=24000)
    assert osr == 24000

def test_limits_peak_when_loud():
    # A loud target pushes the normalized peak above -1 dBFS, so the limiter branch
    # (now applied AFTER resampling) actually fires and must bound the output.
    sr = 48000
    audio = (0.3 * np.sin(2 * np.pi * 3000 * np.arange(sr) / sr)).astype(np.float32)
    out, osr = normalize_for_eval(audio, sr, target_lufs=-3.0)
    assert osr == 16000
    assert float(np.max(np.abs(out))) <= 0.95
