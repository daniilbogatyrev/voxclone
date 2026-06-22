from voxclone.prep.manifest import ClipRecord, write_manifest, read_manifest, write_transcript_csv

def test_manifest_roundtrip(tmp_path):
    records = [
        ClipRecord(audio_path="a.wav", text="hello", duration=2.0,
                   transcript_confidence=0.9, clipped_fraction=0.0, category="harvard"),
        ClipRecord(audio_path="b.wav", text="world", duration=3.0,
                   transcript_confidence=0.7, clipped_fraction=0.0, category="expressive"),
    ]
    p = tmp_path / "manifest.jsonl"
    write_manifest(p, records)
    loaded = read_manifest(p)
    assert loaded == records

def test_transcript_csv_has_header_and_rows(tmp_path):
    records = [ClipRecord(audio_path="a.wav", text="hi", duration=1.0,
                          transcript_confidence=0.8, clipped_fraction=0.0, category="harvard")]
    p = tmp_path / "t.csv"
    write_transcript_csv(p, records)
    lines = p.read_text().splitlines()
    assert lines[0] == "audio_path,category,transcript_confidence,text"
    assert lines[1].startswith("a.wav,harvard,0.8,")

def test_clip_record_snr_default_and_explicit():
    from voxclone.prep.manifest import ClipRecord
    r = ClipRecord(audio_path="a.wav", text="hi", duration=2.0,
                   transcript_confidence=0.9, clipped_fraction=0.0, category="harvard")
    assert r.snr_db == 60.0  # default = "assume clean unless measured"
    r2 = ClipRecord(audio_path="a.wav", text="hi", duration=2.0,
                    transcript_confidence=0.9, clipped_fraction=0.0,
                    category="harvard", snr_db=12.5)
    assert r2.snr_db == 12.5
