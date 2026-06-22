import json
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
from voxclone.common.audio import load_audio, save_audio, peak_normalize, clipped_fraction, resample_audio, estimate_snr_db
from voxclone.common.config import PrepConfig
from voxclone.common.logging import get_logger
from voxclone.prep.denoise import denoise_audio
from voxclone.prep.segment import segment_speech, get_speech_segments
from voxclone.prep.transcribe import transcribe_clip
from voxclone.prep.validate import ValidationThresholds, validate_clip
from voxclone.prep.manifest import ClipRecord, write_manifest, write_transcript_csv

log = get_logger("prep")

@dataclass
class PrepResult:
    kept: list[ClipRecord] = field(default_factory=list)
    quarantined: list[dict] = field(default_factory=list)

def _category_of(stem: str) -> str:
    return stem.split("_")[0] if "_" in stem else "freeform"

def run_prep(raw_dir, out_dir, config: PrepConfig, transcriber,
             vad=None, segmenter=get_speech_segments,
             manifest_path="data/manifest.jsonl",
             transcript_csv="data/transcripts.csv",
             quarantine_report="reports/quarantine.json") -> PrepResult:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = ValidationThresholds(**config.validation.model_dump())
    result = PrepResult()

    for master in sorted(raw_dir.glob("*.wav")):
        audio, sr = load_audio(master)
        audio = resample_audio(audio, sr, config.target_sample_rate)
        sr = config.target_sample_rate
        audio = denoise_audio(audio, sr, enabled=config.denoise_enabled)
        spans = segmenter(audio, sr, vad)
        clips = segment_speech(spans, config.vad_min_duration_s, config.vad_max_duration_s)
        for i, (start, end) in enumerate(clips):
            seg = audio[int(start * sr):int(end * sr)]
            clip_frac = clipped_fraction(seg)
            seg = peak_normalize(seg, config.peak_dbfs)
            clip_path = out_dir / f"{master.stem}_seg{i:03d}.wav"
            save_audio(clip_path, seg, sr)
            tr = transcribe_clip(seg, sr, transcriber)
            record = ClipRecord(
                audio_path=str(clip_path), text=tr.text,
                duration=len(seg) / sr, transcript_confidence=tr.confidence,
                clipped_fraction=clip_frac, category=_category_of(master.stem),
                snr_db=estimate_snr_db(seg, sr),
            )
            reasons = validate_clip(record, thresholds)
            if reasons:
                result.quarantined.append({"audio_path": str(clip_path), "reasons": reasons})
            else:
                result.kept.append(record)

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(transcript_csv).parent.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, result.kept)
    write_transcript_csv(transcript_csv, result.kept)
    Path(quarantine_report).parent.mkdir(parents=True, exist_ok=True)
    Path(quarantine_report).write_text(json.dumps(result.quarantined, indent=2))
    log.info("prep done: kept=%d quarantined=%d", len(result.kept), len(result.quarantined))
    return result
