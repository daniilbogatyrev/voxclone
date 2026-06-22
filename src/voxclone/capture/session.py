"""Guided dataset-recording session — pure, hardware-free core.

Walks the prompt bank in a sensible recording order, gates each recorded clip on the
SAME rules the prep pipeline enforces (3-11 s, SNR >= 30 dB, low clipping), auto-names
clips ``<category>_<index>.wav``, and builds a prep-ready manifest that pairs each clip
with its KNOWN transcript (the prompt text, confidence 1.0) so prep skips Whisper ASR.

Everything here is deterministic and unit-tested. The live ALSA ``arecord`` capture +
keypress loop live in ``scripts/capture.py`` (interactive/hardware, ``# pragma: no cover``)
and call into these helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from voxclone.capture.prompts import PROMPTS
from voxclone.capture.recorder import clip_path, clip_stats
from voxclone.common.audio import estimate_snr_db, load_audio
from voxclone.prep.manifest import ClipRecord, read_manifest, write_manifest

# Record highest-prosodic-value prompts first, then Harvard for phonetic coverage.
CAPTURE_ORDER = ("expressive", "conversational", "technical", "harvard")

# Acceptance gates — match the prep pipeline (clip cap 3-11 s, SNR floor, clipping cap).
MIN_SECONDS = 3.0
MAX_SECONDS = 11.0
MIN_SNR_DB = 30.0
MAX_CLIPPED_FRACTION = 0.01


@dataclass(frozen=True)
class Prompt:
    category: str
    index: int  # 1-based within its category (drives the clip filename)
    text: str


def ordered_prompts(order=CAPTURE_ORDER) -> list[Prompt]:
    """All prompts flattened in recording order, 1-based index per category."""
    return [
        Prompt(cat, i, text)
        for cat in order
        for i, text in enumerate(PROMPTS[cat], start=1)
    ]


def remaining_prompts(raw_dir, order=CAPTURE_ORDER) -> list[Prompt]:
    """Prompts whose clip file does not exist yet — makes a session resumable."""
    raw = Path(raw_dir)
    return [p for p in ordered_prompts(order)
            if not clip_path(raw, p.category, p.index).exists()]


@dataclass
class ClipEval:
    ok: bool
    duration: float
    snr_db: float
    peak: float
    clipped_fraction: float
    reasons: list[str] = field(default_factory=list)  # why rejected; empty when ok


def evaluate_clip(audio, sr, *, snr_fn=estimate_snr_db,
                  min_s: float = MIN_SECONDS, max_s: float = MAX_SECONDS,
                  min_snr: float = MIN_SNR_DB,
                  max_clip: float = MAX_CLIPPED_FRACTION) -> ClipEval:
    """Gate one recorded clip. ``snr_fn`` is injectable so tests stay deterministic."""
    stats = clip_stats(audio, sr)          # {duration_s, peak, clipped_fraction}
    dur = float(stats["duration_s"])
    snr = float(snr_fn(audio, sr))
    clipped = float(stats["clipped_fraction"])
    reasons: list[str] = []
    if dur < min_s:
        reasons.append(f"too short ({dur:.1f}s < {min_s:g}s)")
    elif dur > max_s:
        reasons.append(f"too long ({dur:.1f}s > {max_s:g}s)")
    if snr < min_snr:
        reasons.append(f"noisy (SNR {snr:.1f} dB < {min_snr:g} dB)")
    if clipped > max_clip:
        reasons.append(f"clipped ({clipped * 100:.1f}% of samples)")
    return ClipEval(ok=not reasons, duration=dur, snr_db=snr,
                    peak=float(stats["peak"]), clipped_fraction=clipped, reasons=reasons)


def to_record(prompt: Prompt, audio_path, ev: ClipEval) -> ClipRecord:
    """A prep manifest entry. The transcript IS the prompt (confidence 1.0 — no ASR)."""
    return ClipRecord(
        audio_path=str(audio_path),
        text=prompt.text,
        duration=ev.duration,
        transcript_confidence=1.0,
        clipped_fraction=ev.clipped_fraction,
        category=prompt.category,
        snr_db=ev.snr_db,
    )


def accept_clip(prompt: Prompt, raw_dir, *, snr_fn=estimate_snr_db, **gates):
    """Load the just-recorded clip for ``prompt`` from disk, gate it, and (if it passes)
    build its ClipRecord. Returns ``(ClipEval, ClipRecord | None)``."""
    path = clip_path(raw_dir, prompt.category, prompt.index)
    audio, sr = load_audio(path)
    ev = evaluate_clip(audio, sr, snr_fn=snr_fn, **gates)
    return ev, (to_record(prompt, path, ev) if ev.ok else None)


def upsert(records: list[ClipRecord], record: ClipRecord) -> list[ClipRecord]:
    """Replace any existing record for the same audio_path (a re-take), else append."""
    out = [r for r in records if r.audio_path != record.audio_path]
    out.append(record)
    return out


def load_records(manifest_path) -> list[ClipRecord]:
    """Existing manifest (for resume) or [] if none yet."""
    return read_manifest(manifest_path) if Path(manifest_path).exists() else []


def save_records(manifest_path, records: list[ClipRecord]) -> None:
    write_manifest(manifest_path, records)


def arecord_cmd(device: str, sr: int, out_path) -> list[str]:
    """ALSA capture command: 48 kHz (or ``sr``) mono S24 from ``device`` (e.g. plughw:1,0).
    Run as push-to-stop (start, then terminate the process to end the clip)."""
    return ["arecord", "-D", device, "-f", "S24_LE", "-r", str(sr), "-c", "1", str(out_path)]
