import numpy as np
from voxclone.common.audio import clipped_fraction, peak_normalize, measure_lufs, normalize_lufs

def test_clipped_fraction_counts_samples_at_or_above_threshold():
    audio = np.array([0.0, 1.0, -1.0, 0.5, 0.99], dtype=np.float32)
    frac = clipped_fraction(audio, threshold=0.99)
    assert abs(frac - 3 / 5) < 1e-6  # 1.0, -1.0, 0.99

def test_peak_normalize_sets_peak_to_target_dbfs():
    audio = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = peak_normalize(audio, target_dbfs=-1.0)
    target_peak = 10 ** (-1.0 / 20)
    assert abs(np.max(np.abs(out)) - target_peak) < 1e-4

def test_normalize_lufs_moves_loudness_toward_target():
    sr = 48000
    audio = (0.05 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    out = normalize_lufs(audio, sr, target_lufs=-16.0)
    assert abs(measure_lufs(out, sr) - (-16.0)) < 1.0
