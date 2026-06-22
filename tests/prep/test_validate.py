from voxclone.prep.validate import ValidationThresholds, validate_clip
from voxclone.prep.manifest import ClipRecord

def rec(**kw):
    base = dict(audio_path="x.wav", text="hi there", duration=5.0,
                transcript_confidence=0.9, clipped_fraction=0.0, category="harvard")
    base.update(kw)
    return ClipRecord(**base)

def test_valid_clip_has_no_reasons():
    assert validate_clip(rec(), ValidationThresholds()) == []

def test_clipped_audio_flagged():
    reasons = validate_clip(rec(clipped_fraction=0.05),
                            ValidationThresholds(max_clipped_fraction=0.005))
    assert "clipped" in reasons

def test_too_short_flagged():
    reasons = validate_clip(rec(duration=0.5), ValidationThresholds(min_duration_s=1.5))
    assert "too_short" in reasons

def test_low_confidence_flagged():
    reasons = validate_clip(rec(transcript_confidence=0.2),
                            ValidationThresholds(min_transcript_confidence=0.6))
    assert "low_confidence" in reasons

def test_empty_text_flagged():
    assert "empty_text" in validate_clip(rec(text="  "), ValidationThresholds())

def test_low_snr_flagged():
    reasons = validate_clip(rec(snr_db=12.0), ValidationThresholds(min_snr_db=30.0))
    assert "low_snr" in reasons

def test_too_long_flagged_at_11s_cap():
    reasons = validate_clip(rec(duration=12.0), ValidationThresholds(max_duration_s=11.0))
    assert "too_long" in reasons

def test_text_too_long_flagged():
    reasons = validate_clip(rec(text="x " * 150), ValidationThresholds(max_text_chars=200))
    assert "text_too_long" in reasons

def test_clean_clip_with_good_snr_passes():
    assert validate_clip(rec(snr_db=45.0, duration=8.0), ValidationThresholds()) == []
