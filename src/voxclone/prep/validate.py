from pydantic import BaseModel
from voxclone.prep.manifest import ClipRecord

class ValidationThresholds(BaseModel):
    min_duration_s: float = 3.0
    max_duration_s: float = 11.0
    max_clipped_fraction: float = 0.005
    min_transcript_confidence: float = 0.6
    min_snr_db: float = 30.0
    max_text_chars: int = 200

def validate_clip(record: ClipRecord, thresholds: ValidationThresholds) -> list[str]:
    reasons: list[str] = []
    if record.duration < thresholds.min_duration_s:
        reasons.append("too_short")
    if record.duration > thresholds.max_duration_s:
        reasons.append("too_long")
    if record.clipped_fraction > thresholds.max_clipped_fraction:
        reasons.append("clipped")
    if record.transcript_confidence < thresholds.min_transcript_confidence:
        reasons.append("low_confidence")
    if record.snr_db < thresholds.min_snr_db:
        reasons.append("low_snr")
    if len(record.text) > thresholds.max_text_chars:
        reasons.append("text_too_long")
    if not record.text.strip():
        reasons.append("empty_text")
    return reasons
