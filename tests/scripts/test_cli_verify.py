"""voxclone-verify CLI arg-parsing contract (GPU-free)."""
import scripts.verify_recording as v
from voxclone.capture.verify import DEFAULT_THRESHOLD


def test_parser_defaults():
    args = v.build_parser().parse_args([])
    assert args.manifest.endswith("manifest.jsonl")
    assert args.device == "cuda" and args.model == "large-v3"
    assert args.threshold == DEFAULT_THRESHOLD
    assert args.show_ok is False and args.out is None


def test_parser_overrides():
    args = v.build_parser().parse_args(
        ["--manifest", "/tmp/m.jsonl", "--device", "cpu", "--threshold", "0.5",
         "--show-ok", "--out", "/tmp/flagged.txt"])
    assert args.manifest == "/tmp/m.jsonl" and args.device == "cpu"
    assert args.threshold == 0.5 and args.show_ok is True and args.out == "/tmp/flagged.txt"
