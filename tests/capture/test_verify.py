"""Tests for recording QC (transcript vs script line). ASR is faked — GPU-free."""
import numpy as np

from voxclone.capture import verify as V
from voxclone.capture.recorder import clip_path
from voxclone.common.audio import save_audio
from voxclone.prep.manifest import ClipRecord


def _rec(tmp_path, category, index, text):
    p = clip_path(tmp_path, category, index)
    save_audio(p, np.zeros(16000, dtype=np.float32), 16000)  # content irrelevant; ASR is faked
    return ClipRecord(audio_path=str(p), text=text, duration=1.0,
                      transcript_confidence=1.0, clipped_fraction=0.0, category=category)


def test_compare_correct_read_passes():
    c = V.compare("the cpu runs at 4.2 gigahertz", "the CPU runs at 4.2 gigahertz")
    assert c["ok"] and c["wer"] < 0.1


def test_compare_wrong_sentence_flagged():
    c = V.compare("congratulations you absolutely earned this",
                  "the weather is cold today and nothing matches")
    assert not c["ok"] and c["wer"] > 0.5


def test_compare_threshold_is_tunable():
    # a single dropped word is a small WER; a strict threshold flags it, a loose one doesn't
    strict = V.compare("please lock the front door now", "please lock the door now", threshold=0.05)
    loose = V.compare("please lock the front door now", "please lock the door now", threshold=0.5)
    assert not strict["ok"] and loose["ok"]


def test_verify_records_flags_only_the_misread(tmp_path):
    recs = [
        _rec(tmp_path, "expressive", 1, "I am so proud of you, you have no idea."),
        _rec(tmp_path, "technical", 1, "The package weighs 3.6 kilograms and ships within 48 hours."),
    ]
    # fake ASR: clip 1 read correctly, clip 2 read the WRONG line
    heard = {
        recs[0].audio_path: "I am so proud of you you have no idea",
        recs[1].audio_path: "let us get out of here before it gets dark",
    }
    # transcribe_fn gets (audio, sr); map back via call order using a closure over an index
    calls = {"i": 0}
    def fake_transcribe(audio, sr):
        path = recs[calls["i"]].audio_path
        calls["i"] += 1
        assert sr == V.ASR_SR  # resampled to 16k for ASR
        return heard[path]

    rep = V.verify_records(recs, transcribe_fn=fake_transcribe)
    assert rep["summary"]["total"] == 2
    assert rep["summary"]["flagged"] == 1
    assert rep["flagged"][0]["audio_path"] == recs[1].audio_path
    assert rep["flagged"][0]["wer"] > 0.5


def test_verify_records_resamples_to_16k(tmp_path):
    # a 48 kHz clip must be handed to the ASR at 16 kHz
    p = clip_path(tmp_path, "harvard", 1)
    save_audio(p, np.zeros(48000, dtype=np.float32), 48000)
    rec = ClipRecord(audio_path=str(p), text="hello world", duration=1.0,
                     transcript_confidence=1.0, clipped_fraction=0.0, category="harvard")
    seen = {}
    def fake_transcribe(audio, sr):
        seen["sr"], seen["n"] = sr, len(audio)
        return "hello world"
    V.verify_records([rec], transcribe_fn=fake_transcribe)
    assert seen["sr"] == 16000 and seen["n"] == 16000  # 48k -> 16k


def test_format_report_lists_only_flagged_by_default(tmp_path):
    recs = [_rec(tmp_path, "harvard", 1, "good morning"),
            _rec(tmp_path, "harvard", 2, "the quick brown fox")]
    order = iter(["good morning", "completely wrong words entirely"])  # clip 1 ok, clip 2 misread
    rep = V.verify_records(recs, transcribe_fn=lambda a, s: next(order))
    out = V.format_report(rep)
    assert "1 flagged" in out and "harvard_0002.wav" in out and "harvard_0001.wav" not in out
