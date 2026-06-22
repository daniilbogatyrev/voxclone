import numpy as np
from voxclone.capture.recorder import clip_path, clip_stats, SessionTotals

def test_clip_path_format(tmp_path):
    p = clip_path(tmp_path, "harvard", 3)
    assert p.name == "harvard_0003.wav"
    assert p.parent == tmp_path

def test_clip_stats_reports_peak_and_clipping():
    audio = np.array([0.0, 1.0, -0.5], dtype=np.float32)
    stats = clip_stats(audio, sr=48000)
    assert abs(stats["duration_s"] - 3 / 48000) < 1e-6
    assert abs(stats["peak"] - 1.0) < 1e-6
    assert stats["clipped_fraction"] > 0.0

def test_session_totals_accumulate():
    totals = SessionTotals()
    totals.add(duration_s=2.0)
    totals.add(duration_s=3.0)
    assert totals.clip_count == 2
    assert abs(totals.total_minutes - 5 / 60) < 1e-6

def test_calibration_sentence_is_nonempty_constant():
    from voxclone.capture.recorder import CALIBRATION_SENTENCE
    assert isinstance(CALIBRATION_SENTENCE, str) and len(CALIBRATION_SENTENCE) > 20

def test_session_cap_reached():
    from voxclone.capture.recorder import session_cap_reached, SessionTotals
    t = SessionTotals()
    t.add(duration_s=19 * 60)
    assert session_cap_reached(t, max_minutes=20.0) is False
    t.add(duration_s=2 * 60)
    assert session_cap_reached(t, max_minutes=20.0) is True
