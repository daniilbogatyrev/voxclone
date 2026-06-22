"""Tests for the guided dataset-recording session core (hardware-free)."""
import numpy as np
import pytest

from voxclone.capture import session as S
from voxclone.capture.recorder import clip_path
from voxclone.common.audio import save_audio
from voxclone.prep.manifest import ClipRecord, read_manifest


SR = 48000


def _wav(seconds: float, amp: float = 0.3) -> np.ndarray:
    """A non-silent test clip: a low-noise tone, mono float32."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def _hi(audio, sr):   # injected SNR seam -> clean
    return 45.0


def _lo(audio, sr):   # injected SNR seam -> noisy
    return 18.0


# ---- prompt ordering / resume -------------------------------------------------
def test_ordered_prompts_full_bank_in_capture_order():
    ps = S.ordered_prompts()
    assert len(ps) == sum(len(v) for v in S.PROMPTS.values())   # 814
    assert ps[0].category == "expressive" and ps[0].index == 1
    assert ps[-1].category == "harvard"
    # index is 1-based per category
    expr = [p for p in ps if p.category == "expressive"]
    assert [p.index for p in expr[:3]] == [1, 2, 3]
    assert expr[0].text == S.PROMPTS["expressive"][0]


def test_remaining_prompts_skips_existing_clips(tmp_path):
    assert len(S.remaining_prompts(tmp_path)) == len(S.ordered_prompts())
    # "record" the first expressive clip -> it drops out of the remaining list
    p0 = S.ordered_prompts()[0]
    save_audio(clip_path(tmp_path, p0.category, p0.index), _wav(4), SR)
    rem = S.remaining_prompts(tmp_path)
    assert len(rem) == len(S.ordered_prompts()) - 1
    assert not any(r.category == p0.category and r.index == p0.index for r in rem)


# ---- clip evaluation gates ----------------------------------------------------
def test_evaluate_clip_accepts_clean_in_range():
    ev = S.evaluate_clip(_wav(5), SR, snr_fn=_hi)
    assert ev.ok and ev.reasons == []
    assert 4.9 < ev.duration < 5.1 and ev.snr_db == 45.0


def test_evaluate_clip_rejects_too_short():
    ev = S.evaluate_clip(_wav(1.5), SR, snr_fn=_hi)
    assert not ev.ok and any("too short" in r for r in ev.reasons)


def test_evaluate_clip_rejects_too_long():
    ev = S.evaluate_clip(_wav(13), SR, snr_fn=_hi)
    assert not ev.ok and any("too long" in r for r in ev.reasons)


def test_evaluate_clip_rejects_noisy():
    ev = S.evaluate_clip(_wav(5), SR, snr_fn=_lo)
    assert not ev.ok and any("noisy" in r for r in ev.reasons)


def test_evaluate_clip_rejects_clipped():
    clipped = np.ones(SR * 5, dtype=np.float32)        # full-scale -> clipped_fraction ~1
    ev = S.evaluate_clip(clipped, SR, snr_fn=_hi)
    assert not ev.ok and any("clipped" in r for r in ev.reasons)


# ---- record building / manifest ----------------------------------------------
def test_to_record_uses_known_transcript_confidence_one():
    p = S.Prompt("technical", 4, "The CPU runs at 4.2 gigahertz across all 16 cores.")
    ev = S.evaluate_clip(_wav(5), SR, snr_fn=_hi)
    rec = S.to_record(p, "/x/technical_0004.wav", ev)
    assert isinstance(rec, ClipRecord)
    assert rec.text == p.text and rec.category == "technical"
    assert rec.transcript_confidence == 1.0          # known transcript -> no Whisper
    assert rec.snr_db == 45.0 and rec.audio_path.endswith("technical_0004.wav")


def test_accept_clip_round_trip_to_manifest(tmp_path):
    p = S.Prompt("conversational", 1, S.PROMPTS["conversational"][0])
    save_audio(clip_path(tmp_path, p.category, p.index), _wav(5), SR)
    ev, rec = S.accept_clip(p, tmp_path, snr_fn=_hi)
    assert ev.ok and rec is not None
    records = S.upsert(S.load_records(tmp_path / "m.jsonl"), rec)
    S.save_records(tmp_path / "m.jsonl", records)
    back = read_manifest(tmp_path / "m.jsonl")        # prep can read it
    assert len(back) == 1 and back[0].text == p.text and back[0].transcript_confidence == 1.0


def test_accept_clip_returns_none_record_when_rejected(tmp_path):
    p = S.Prompt("expressive", 1, S.PROMPTS["expressive"][0])
    save_audio(clip_path(tmp_path, p.category, p.index), _wav(1.0), SR)   # too short
    ev, rec = S.accept_clip(p, tmp_path, snr_fn=_hi)
    assert not ev.ok and rec is None


def test_upsert_replaces_same_path_on_retake():
    a = S.to_record(S.Prompt("harvard", 1, "one"), "/x/harvard_0001.wav",
                    S.ClipEval(True, 4.0, 40.0, 0.3, 0.0))
    b = S.to_record(S.Prompt("harvard", 1, "one"), "/x/harvard_0001.wav",
                    S.ClipEval(True, 5.0, 42.0, 0.3, 0.0))
    out = S.upsert([a], b)
    assert len(out) == 1 and out[0].duration == 5.0    # the re-take won


# ---- arecord command ----------------------------------------------------------
def test_arecord_cmd_is_mono_s24_at_rate():
    cmd = S.arecord_cmd("plughw:1,0", 48000, "/tmp/x.wav")
    assert cmd[:3] == ["arecord", "-D", "plughw:1,0"]
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "1"        # mono
    assert cmd[cmd.index("-r") + 1] == "48000"
    assert "S24_LE" in cmd and cmd[-1] == "/tmp/x.wav"
