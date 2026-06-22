"""voxclone-import CLI arg-parsing contract (GPU-free)."""
import scripts.import_recording as imp


def test_parser_defaults():
    a = imp.build_parser().parse_args([])
    assert a.input.endswith("full_recording.wav")
    assert a.raw_dir.endswith("raw") and a.manifest.endswith("manifest.jsonl")
    assert a.device == "cuda" and a.model == "large-v3"
    assert a.min_s == 3.0 and a.max_s == 11.0


def test_parser_overrides():
    a = imp.build_parser().parse_args(
        ["--input", "/tmp/take.wav", "--device", "cpu", "--snr-gate", "30", "--max-s", "9"])
    assert a.input == "/tmp/take.wav" and a.device == "cpu"
    assert a.snr_gate == 30.0 and a.max_s == 9.0
