import numpy as np

def segment_speech(spans: list[dict], min_dur: float = 3.0,
                   max_dur: float = 15.0) -> list[tuple[float, float]]:
    """Split each speech span (seconds) into clips within [min_dur, max_dur].
    Spans shorter than min_dur are dropped; long spans are evenly chunked."""
    clips: list[tuple[float, float]] = []
    for span in spans:
        start, end = float(span["start"]), float(span["end"])
        length = end - start
        if length < min_dur:
            continue
        n_chunks = max(1, int(np.ceil(length / max_dur)))
        chunk_len = length / n_chunks
        for i in range(n_chunks):
            s = start + i * chunk_len
            e = start + (i + 1) * chunk_len if i < n_chunks - 1 else end
            if (e - s) >= min_dur:
                clips.append((round(s, 6), round(e, 6)))
    return clips

def merge_spans(spans: list[dict], *, min_s: float = 3.0, max_s: float = 11.0,
                max_gap: float = 1.5) -> list[tuple[float, float]]:
    """Merge CONSECUTIVE VAD spans into clips within [min_s, max_s].

    Unlike ``segment_speech`` (which DROPS spans shorter than min_dur), this grows a clip
    by appending the next span as long as the clip stays <= max_s and the inter-span gap is
    <= max_gap (so the brief pauses between sentences are kept inside the clip — a natural,
    contiguous cut). A new clip starts when the next span would overflow max_s or sits after
    a gap > max_gap. Clips that still end up shorter than min_s are absorbed into the
    previous clip when adjacent, else dropped. This is the right tool when silero over-
    segments at intra-sentence micro-pauses (median span ~2 s)."""
    clips: list[tuple[float, float]] = []
    cur: list[float] | None = None
    for sp in spans:
        s, e = float(sp["start"]), float(sp["end"])
        if cur is None:
            cur = [s, e]
        elif (e - cur[0]) <= max_s and (s - cur[1]) <= max_gap:
            cur[1] = e
        else:
            clips.append((cur[0], cur[1]))
            cur = [s, e]
    if cur is not None:
        clips.append((cur[0], cur[1]))

    out: list[tuple[float, float]] = []
    for c in clips:
        if (c[1] - c[0]) >= min_s:
            out.append((round(c[0], 6), round(c[1], 6)))
        elif out and (c[0] - out[-1][1]) <= max_gap and (c[1] - out[-1][0]) <= max_s:
            out[-1] = (out[-1][0], round(c[1], 6))          # absorb a short tail (only if it stays <= max_s)
        # else: a short, isolated clip that can't be absorbed within max_s -> drop (rare)
    return out


def load_vad():  # pragma: no cover (model)
    from silero_vad import load_silero_vad
    return load_silero_vad()

def get_speech_segments(audio: np.ndarray, sr: int, vad) -> list[dict]:  # pragma: no cover
    import torch
    from silero_vad import get_speech_timestamps
    ts = get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=sr)
    return [{"start": t["start"] / sr, "end": t["end"] / sr} for t in ts]
