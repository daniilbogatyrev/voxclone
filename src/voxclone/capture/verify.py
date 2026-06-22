"""Recording QC — confirm each clip actually says its script line.

After recording, the manifest pairs every clip with the sentence you *meant* to read
(``ClipRecord.text``). This re-transcribes each clip with WhisperX and compares the ASR
text to that expected line via the project's Whisper-normalized WER. Clips whose WER
exceeds a threshold are FLAGGED for review/re-record — they're likely a misread, a skip,
the wrong sentence, or a dead/silent take. A correct read normalizes to ~0 WER; a wrong
line is ~1.0.

The ASR itself (WhisperX) is the only heavy/GPU part and lives in the CLI behind an
injectable ``transcribe_fn``; everything here is deterministic and unit-tested.
"""
from __future__ import annotations

from voxclone.common.audio import load_audio, resample_audio
from voxclone.eval.wer import wer as _wer

# Above this normalized WER a clip is flagged. A clean correct read is well under this;
# a wrong/very-misread sentence is far above it. Tune via the CLI --threshold.
DEFAULT_THRESHOLD = 0.30
ASR_SR = 16000  # WhisperX expects 16 kHz mono


def _to_asr_audio(audio, sr, load=load_audio, target_sr: int = ASR_SR):
    """Mono, 16 kHz float — what the ASR wants."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def compare(expected_text: str, asr_text: str, *, threshold: float = DEFAULT_THRESHOLD,
            wer_fn=_wer) -> dict:
    """WER between the intended line and what was heard; ok if within threshold."""
    w = float(wer_fn(expected_text, asr_text))
    return {"wer": w, "ok": w <= threshold, "expected": expected_text, "hyp": asr_text}


def verify_records(records, *, transcribe_fn, threshold: float = DEFAULT_THRESHOLD,
                   wer_fn=_wer, load=load_audio) -> dict:
    """Transcribe + score every ClipRecord. ``transcribe_fn(audio, sr) -> str`` is the
    injectable ASR seam (the CLI wires WhisperX; tests pass a fake)."""
    results = []
    for r in records:
        audio, sr = load(r.audio_path)
        audio, sr = _to_asr_audio(audio, sr, load=load)
        hyp = transcribe_fn(audio, sr)
        c = compare(r.text, hyp, threshold=threshold, wer_fn=wer_fn)
        results.append({"audio_path": r.audio_path, "category": getattr(r, "category", ""),
                        **c})
    flagged = [x for x in results if not x["ok"]]
    mean_wer = sum(x["wer"] for x in results) / len(results) if results else 0.0
    return {
        "results": results,
        "flagged": flagged,
        "summary": {"total": len(results), "flagged": len(flagged),
                    "mean_wer": mean_wer},
    }


def format_report(report: dict, *, show_ok: bool = False) -> str:
    """Human-readable QC report, worst (highest WER) first."""
    s = report["summary"]
    lines = [f"Recording QC: {s['total']} clips, {s['flagged']} flagged "
             f"(mean WER {s['mean_wer']:.2f}, threshold-based).", ""]
    rows = sorted(report["results"], key=lambda x: -x["wer"])
    for x in rows:
        if x["ok"] and not show_ok:
            continue
        mark = "✓" if x["ok"] else "⚠"
        lines.append(f"{mark} WER {x['wer']:.2f}  {x['audio_path']}")
        if not x["ok"]:
            lines.append(f"    script: {x['expected']}")
            lines.append(f"    heard : {x['hyp']}")
    if not report["flagged"]:
        lines.append("✓ No misreads detected — every clip matches its script line.")
    return "\n".join(lines)
