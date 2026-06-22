"""Turn a long single-speaker take into per-clip ClipRecords.

Given silero-VAD speech spans (already split to 3–11 s by ``segment.segment_speech``),
cut each from the original-rate audio, gate it (SNR / duration / clipping), transcribe it
(injected ``transcribe_fn``), save ``<prefix>_NNNN.wav`` and build a ClipRecord. The VAD +
WhisperX heavy lifting live in scripts/import_recording.py; this core is deterministic and
unit-tested with a fake transcriber.
"""
from __future__ import annotations

from pathlib import Path

from voxclone.common.audio import clipped_fraction, estimate_snr_db, save_audio
from voxclone.prep.manifest import ClipRecord

DEFAULT_SNR_GATE = 26.0   # accept clips at/above this dB (set a touch below 30 by choice)
MIN_S = 3.0
MAX_S = 11.0
MAX_CLIPPED = 0.01


def build_records(clip_spans, audio, sr: int, raw_dir, *, transcribe_fn,
                  snr_gate: float = DEFAULT_SNR_GATE, min_s: float = MIN_S,
                  max_s: float = MAX_S, category: str = "read", prefix: str = "read"):
    """Cut + gate + transcribe each (start, end) span. ``transcribe_fn(clip, sr) ->
    (text, confidence)``. Returns ``(records, flagged)`` where flagged lists clips that
    fail a gate (still recorded, so you can review/re-take). Spans far below min_s are
    skipped entirely."""
    raw = Path(raw_dir).resolve()   # absolute: F5/XTTS formatters require absolute wav paths
    raw.mkdir(parents=True, exist_ok=True)
    records: list[ClipRecord] = []
    flagged: list[dict] = []
    n = 0
    for (s, e) in clip_spans:
        clip = audio[int(float(s) * sr):int(float(e) * sr)]
        dur = len(clip) / sr
        if dur < min_s * 0.8 or clip.size == 0:   # too short to be a usable utterance
            continue
        snr = float(estimate_snr_db(clip, sr))
        cfrac = float(clipped_fraction(clip))
        text, conf = transcribe_fn(clip, sr)
        text = (text or "").strip()
        n += 1
        path = raw / f"{prefix}_{n:04d}.wav"
        save_audio(path, clip, sr)
        records.append(ClipRecord(
            audio_path=str(path), text=text, duration=float(dur),
            transcript_confidence=float(conf), clipped_fraction=cfrac,
            category=category, snr_db=snr))
        reasons = []
        if not (min_s <= dur <= max_s):
            reasons.append(f"duration {dur:.1f}s")
        if snr < snr_gate:
            reasons.append(f"SNR {snr:.0f}dB")
        if cfrac > MAX_CLIPPED:
            reasons.append(f"clipping {cfrac*100:.1f}%")
        if not text:
            reasons.append("empty transcript")
        if reasons:
            flagged.append({"path": str(path), "reasons": reasons, "text": text})
    return records, flagged


def summarize(records, flagged, *, snr_gate: float = DEFAULT_SNR_GATE) -> str:
    if not records:
        return "No clips produced."
    snrs = sorted(r.snr_db for r in records)
    durs = [r.duration for r in records]
    below = sum(1 for r in records if r.snr_db < snr_gate)
    below30 = sum(1 for r in records if r.snr_db < 30)
    total_min = sum(durs) / 60
    lines = [
        f"{len(records)} clips · {total_min:.1f} min total · {len(flagged)} flagged.",
        f"duration: {min(durs):.1f}–{max(durs):.1f}s (mean {sum(durs)/len(durs):.1f}s).",
        f"SNR: min {snrs[0]:.0f} / median {snrs[len(snrs)//2]:.0f} / max {snrs[-1]:.0f} dB · "
        f"{below30} below 30 · {below} below the {snr_gate:.0f} gate.",
    ]
    return "\n".join(lines)
