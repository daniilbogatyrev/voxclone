"""voxclone-verify — confirm each recorded clip actually says its script line.

Transcribes every clip in the manifest with WhisperX and flags any whose ASR text
diverges from the sentence you meant to read (misread / skip / wrong line / silent take),
so the "transcript == prompt" assumption the training relies on actually holds.

    voxclone-verify --manifest experiments/danil/manifest.jsonl --device cuda

Review the flagged clips, re-record them (voxclone-capture resumes), and re-run.
"""
from __future__ import annotations

import argparse
import sys

from voxclone.capture.verify import DEFAULT_THRESHOLD, format_report, verify_records
from voxclone.prep.manifest import read_manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voxclone-verify", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="experiments/danil/manifest.jsonl",
                   help="manifest.jsonl of recorded clips (ClipRecord per line)")
    p.add_argument("--device", default="cuda", help="WhisperX device (cuda/cpu)")
    p.add_argument("--model", default="large-v3", help="WhisperX model name")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="normalized-WER above which a clip is flagged as a likely misread")
    p.add_argument("--show-ok", action="store_true", help="also list clips that passed")
    p.add_argument("--out", default=None, help="optional file to write flagged audio_paths to")
    return p


def run(args) -> int:  # pragma: no cover (loads WhisperX / GPU)
    from voxclone.prep.transcribe import load_transcriber, transcribe_clip

    records = read_manifest(args.manifest)
    if not records:
        print(f"No records in {args.manifest} — nothing to verify.")
        return 0
    print(f"Transcribing {len(records)} clips with WhisperX {args.model} on {args.device}…")
    model = load_transcriber(args.model, args.device)

    def transcribe_fn(audio, sr):
        return transcribe_clip(audio, sr, model).text

    report = verify_records(records, transcribe_fn=transcribe_fn, threshold=args.threshold)
    print(format_report(report, show_ok=args.show_ok))
    if args.out and report["flagged"]:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(x["audio_path"] for x in report["flagged"]) + "\n")
        print(f"\nFlagged paths written to {args.out} (re-record these, then re-run).")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
