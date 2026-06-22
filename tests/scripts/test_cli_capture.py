"""voxclone-capture CLI arg-parsing contract (GPU/hardware-free)."""
import scripts.capture as cap


def test_parser_defaults():
    args = cap.build_parser().parse_args([])
    assert args.device == "plughw:1,0"
    assert args.raw_dir.endswith("raw") and args.manifest.endswith(".jsonl")
    assert args.sr == 48000 and args.session_minutes == 20.0
    assert args.order == ["expressive", "conversational", "technical", "harvard"]
    assert args.list_devices is False


def test_parser_overrides():
    args = cap.build_parser().parse_args(
        ["--device", "plughw:2,0", "--sr", "44100", "--session-minutes", "15",
         "--raw-dir", "/tmp/r", "--manifest", "/tmp/m.jsonl", "--list-devices"])
    assert args.device == "plughw:2,0" and args.sr == 44100
    assert args.session_minutes == 15.0 and args.raw_dir == "/tmp/r"
    assert args.manifest == "/tmp/m.jsonl" and args.list_devices is True
