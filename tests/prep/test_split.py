from voxclone.prep.manifest import ClipRecord
from voxclone.prep.split import split_dataset, write_heldout_tsv, SplitResult

def _rec(i, cat="harvard", text=None, dur=5.0, snr=45.0):
    return ClipRecord(audio_path=f"clip_{i}.wav", text=text or f"sentence number {i}",
                      duration=dur, transcript_confidence=0.9, clipped_fraction=0.0,
                      category=cat, snr_db=snr)

def _dataset(n=60):
    cats = ["harvard", "expressive", "conversational", "technical"]
    recs = [_rec(i, cat=cats[i % 4]) for i in range(n)]
    for i in range(4):
        recs.append(_rec(1000 + i, cat="conversational", dur=8.0, snr=55.0))
    return recs

def test_split_sizes_and_disjoint():
    res = split_dataset(_dataset(), n_heldout=20, n_enrollment=4, seed=0)
    assert isinstance(res, SplitResult)
    assert len(res.enrollment) == 4
    assert len(res.held_out) == 20
    ids = lambda rs: {r.audio_path for r in rs}
    assert ids(res.train) & ids(res.held_out) == set()
    assert ids(res.train) & ids(res.enrollment) == set()
    assert ids(res.held_out) & ids(res.enrollment) == set()

def test_enrollment_within_duration_window():
    res = split_dataset(_dataset(), seed=0, enrollment_min_s=6.0, enrollment_max_s=10.0)
    assert all(6.0 <= r.duration <= 10.0 for r in res.enrollment)

def test_deterministic_with_seed():
    a = split_dataset(_dataset(), seed=7)
    b = split_dataset(_dataset(), seed=7)
    assert [r.audio_path for r in a.held_out] == [r.audio_path for r in b.held_out]

def test_near_duplicates_do_not_straddle():
    recs = [_rec(i) for i in range(40)]
    recs.append(_rec(900, text="the quick brown fox jumps high"))
    recs.append(_rec(901, text="the quick brown fox jumps high"))
    recs += [_rec(1000 + i, cat="conversational", dur=8.0, snr=55.0) for i in range(4)]
    res = split_dataset(recs, n_heldout=10, seed=0)
    held_ids = {r.audio_path for r in res.held_out}
    train_ids = {r.audio_path for r in res.train}
    dup = {"clip_900.wav", "clip_901.wav"}
    assert not (dup & held_ids and dup & train_ids)

def test_write_heldout_tsv(tmp_path):
    res = split_dataset(_dataset(), n_heldout=5, seed=0)
    p = tmp_path / "held.tsv"
    write_heldout_tsv(p, res.held_out)
    lines = p.read_text().splitlines()
    assert len(lines) == 5
    assert all("\t" in ln for ln in lines)

def test_write_enrollment(tmp_path):
    from voxclone.prep.split import write_enrollment
    res = split_dataset(_dataset(), n_enrollment=4, seed=0)
    p = tmp_path / "enroll.tsv"
    write_enrollment(p, res.enrollment)
    lines = p.read_text().splitlines()
    assert len(lines) == 4
    for ln, r in zip(lines, res.enrollment):  # path<TAB>text order
        path, text = ln.split("\t", 1)
        assert path == r.audio_path and text == r.text
