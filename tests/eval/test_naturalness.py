import numpy as np
from voxclone.eval.naturalness import aggregate_utmos, utmos_score

def test_aggregate_is_mean():
    assert abs(aggregate_utmos([3.0, 4.0, 5.0]) - 4.0) < 1e-6

def test_aggregate_empty_is_zero():
    assert aggregate_utmos([]) == 0.0

def test_utmos_score_uses_injected_model():
    fake_model = lambda audio, sr: 4.2
    assert utmos_score(np.zeros(100, dtype=np.float32), 16000, fake_model) == 4.2
