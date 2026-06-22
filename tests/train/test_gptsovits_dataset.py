from voxclone.prep.manifest import ClipRecord
from voxclone.train.gptsovits import manifest_to_gptsovits

def test_converter_emits_pipe_format():
    records = [
        ClipRecord(audio_path="data/processed/a.wav", text="Hello world.",
                   duration=2.0, transcript_confidence=0.9,
                   clipped_fraction=0.0, category="harvard"),
    ]
    out = manifest_to_gptsovits(records, speaker="target")
    assert out.strip() == "data/processed/a.wav|target|EN|Hello world."

def test_converter_skips_empty_text():
    records = [
        ClipRecord(audio_path="a.wav", text="  ", duration=2.0,
                   transcript_confidence=0.9, clipped_fraction=0.0, category="x"),
        ClipRecord(audio_path="b.wav", text="ok", duration=2.0,
                   transcript_confidence=0.9, clipped_fraction=0.0, category="x"),
    ]
    lines = manifest_to_gptsovits(records, speaker="t").splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("b.wav|")
