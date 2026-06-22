"""Tests for the long-take importer core (VAD/ASR faked — GPU-free)."""
import numpy as np

from voxclone.prep import importer as I
from voxclone.prep.manifest import ClipRecord, read_manifest, write_manifest

SR = 44100


def _tone(seconds, amp=0.3):
    n = int(seconds * SR)
    return (amp * np.sin(2 * np.pi * 180 * np.arange(n) / SR)).astype(np.float32)


def test_build_records_cuts_gates_saves_and_records(tmp_path):
    audio = _tone(20)
    spans = [(0.0, 5.0), (6.0, 10.0), (11.0, 12.0)]  # 5s, 4s, 1s(too short -> dropped)
    texts = iter(["hello world one", "second clip text"])
    def fake(clip, sr):
        assert sr == SR
        return next(texts), 0.95
    recs, flagged = I.build_records(spans, audio, SR, tmp_path / "raw",
                                    transcribe_fn=fake, snr_gate=0.0)
    import os
    assert len(recs) == 2                                   # the 1s span was skipped
    assert (tmp_path / "raw" / "read_0001.wav").exists()
    assert (tmp_path / "raw" / "read_0002.wav").exists()
    assert os.path.isabs(recs[0].audio_path)                # F5/XTTS need absolute paths
    assert recs[0].text == "hello world one" and recs[0].category == "read"
    assert recs[0].transcript_confidence == 0.95
    assert abs(recs[0].duration - 5.0) < 0.2
    assert flagged == []                                    # snr_gate 0 -> nothing flagged


def test_build_records_flags_below_gate_and_empty_text(tmp_path):
    audio = _tone(12)
    spans = [(0.0, 5.0), (6.0, 10.0)]
    recs, flagged = I.build_records(spans, audio, SR, tmp_path / "raw",
                                    transcribe_fn=lambda c, s: ("", 0.5), snr_gate=99.0)
    assert len(recs) == 2
    # every clip flagged: SNR below the (impossible) 99 dB gate AND empty transcript
    assert len(flagged) == 2
    assert any("SNR" in r for r in flagged[0]["reasons"])
    assert any("empty transcript" in r for r in flagged[0]["reasons"])


def test_records_round_trip_to_manifest(tmp_path):
    audio = _tone(8)
    recs, _ = I.build_records([(0.0, 5.0)], audio, SR, tmp_path / "raw",
                              transcribe_fn=lambda c, s: ("good morning", 0.9), snr_gate=0.0)
    write_manifest(tmp_path / "m.jsonl", recs)
    back = read_manifest(tmp_path / "m.jsonl")
    assert len(back) == 1 and isinstance(back[0], ClipRecord)
    assert back[0].text == "good morning"


def test_summarize_reports_counts(tmp_path):
    audio = _tone(20)
    recs, flagged = I.build_records([(0.0, 5.0), (6.0, 10.0)], audio, SR, tmp_path / "raw",
                                    transcribe_fn=lambda c, s: ("t", 0.9), snr_gate=0.0)
    out = I.summarize(recs, flagged)
    assert "2 clips" in out and "SNR" in out
