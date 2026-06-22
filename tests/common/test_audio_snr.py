import numpy as np
from voxclone.common.audio import estimate_snr_db

def test_snr_high_for_loud_over_quiet():
    sr = 16000
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 0.001, sr // 2).astype(np.float32)
    loud = rng.normal(0, 0.3, sr // 2).astype(np.float32)
    audio = np.concatenate([quiet, loud])
    assert estimate_snr_db(audio, sr) > 30

def test_snr_low_for_uniform_noise():
    sr = 16000
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.05, sr).astype(np.float32)
    assert estimate_snr_db(audio, sr) < 10

def test_snr_empty_is_zero():
    assert estimate_snr_db(np.zeros(0, dtype=np.float32), 16000) == 0.0
