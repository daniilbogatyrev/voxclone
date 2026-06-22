from voxclone.prep.segment import merge_spans, segment_speech

def test_long_span_is_split_under_max():
    spans = [{"start": 0.0, "end": 40.0}]
    out = segment_speech(spans, min_dur=3.0, max_dur=15.0)
    assert all(3.0 <= (e - s) <= 15.0 for s, e in out)
    assert out[0][0] == 0.0
    assert abs(out[-1][1] - 40.0) < 1e-6

def test_short_span_is_dropped():
    spans = [{"start": 0.0, "end": 1.0}]
    out = segment_speech(spans, min_dur=3.0, max_dur=15.0)
    assert out == []

def test_in_range_span_passes_through():
    spans = [{"start": 2.0, "end": 9.0}]
    out = segment_speech(spans, min_dur=3.0, max_dur=15.0)
    assert out == [(2.0, 9.0)]


# --- merge_spans: the fix for silero over-segmentation -------------------------
def _spans(lengths, gap=0.3, start=0.0):
    out, t = [], start
    for L in lengths:
        out.append({"start": round(t, 3), "end": round(t + L, 3)})
        t += L + gap
    return out


def test_segment_speech_drops_short_spans_but_merge_keeps_them():
    # five ~2 s spans (like silero's median 2.24 s) — the real-world failure case
    spans = _spans([2.0] * 5, gap=0.3)
    assert segment_speech(spans, min_dur=3.0, max_dur=11.0) == []   # the BUG: all dropped
    clips = merge_spans(spans, min_s=3.0, max_s=11.0)              # the FIX: merged, kept
    assert clips
    covered = sum(e - s for s, e in clips)
    assert covered > 8.0                                          # ~all of the ~11.2 s retained


def test_merge_spans_clips_within_bounds_and_contiguous():
    spans = _spans([2.0] * 12, gap=0.4)
    clips = merge_spans(spans, min_s=3.0, max_s=11.0, max_gap=1.5)
    assert all(3.0 <= (e - s) <= 11.0 + 1e-6 for s, e in clips)
    # clips are ordered and non-overlapping
    for (s0, e0), (s1, e1) in zip(clips, clips[1:]):
        assert s1 >= e0


def test_merge_spans_absorb_never_exceeds_max():
    # a ~10.5 s clip then a 2 s sliver at a small gap must NOT absorb into a 12.5 s clip
    spans = [{"start": 0.0, "end": 10.5}, {"start": 10.8, "end": 12.8}]
    clips = merge_spans(spans, min_s=3.0, max_s=11.0, max_gap=1.5)
    assert all((e - s) <= 11.0 + 1e-6 for s, e in clips)         # the sliver is dropped, not absorbed


def test_merge_spans_breaks_on_large_gap():
    # a >max_gap pause should END a clip, not be bridged into a long silence
    spans = [{"start": 0.0, "end": 2.5}, {"start": 2.8, "end": 5.0},
             {"start": 30.0, "end": 32.5}, {"start": 32.8, "end": 35.0}]
    clips = merge_spans(spans, min_s=3.0, max_s=11.0, max_gap=1.5)
    assert len(clips) == 2                                        # split at the 25 s gap
    assert clips[0][1] < 6 and clips[1][0] >= 30
