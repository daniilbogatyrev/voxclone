"""voxclone-import — turn a long single-speaker take into clips + a prep-ready manifest.

silero-VAD splits the recording at the pauses into 3–11 s clips, WhisperX transcribes each,
they're gated (SNR / duration / clipping), saved as <prefix>_NNNN.wav, and written to
manifest.jsonl (one ClipRecord per clip, transcript = the ASR text). Then: voxclone-verify,
then prep/split, then fine-tune.

    voxclone-import --input experiments/danil/full_recording.wav \\
                    --raw-dir experiments/danil/raw --manifest experiments/danil/manifest.jsonl
"""
from __future__ import annotations

import argparse
import sys

from voxclone.prep.importer import DEFAULT_SNR_GATE, MAX_S, MIN_S, build_records, summarize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voxclone-import", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="experiments/danil/full_recording.wav",
                   help="the long take (wav/flac/mp3/ogg)")
    p.add_argument("--raw-dir", default="experiments/danil/raw", help="where clips are written")
    p.add_argument("--manifest", default="experiments/danil/manifest.jsonl",
                   help="prep-ready manifest output (JSONL of ClipRecord)")
    p.add_argument("--device", default="cuda", help="WhisperX device")
    p.add_argument("--model", default="large-v3", help="WhisperX model")
    p.add_argument("--snr-gate", type=float, default=DEFAULT_SNR_GATE,
                   help="flag clips below this SNR (dB)")
    p.add_argument("--min-s", type=float, default=MIN_S)
    p.add_argument("--max-s", type=float, default=MAX_S)
    return p


def run(args) -> int:  # pragma: no cover (VAD + WhisperX + GPU)
    import numpy as np

    from voxclone.common.audio import load_audio, resample_audio
    from voxclone.prep.manifest import write_manifest, write_transcript_csv
    from voxclone.prep.segment import get_speech_segments, load_vad, merge_spans
    from voxclone.prep.transcribe import load_transcriber, transcribe_clip

    audio, sr = load_audio(args.input)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    print(f"loaded {len(audio)/sr/60:.1f} min @ {sr} Hz — running silero-VAD…")
    audio16 = resample_audio(audio, sr, 16000) if sr != 16000 else audio
    spans = get_speech_segments(audio16, 16000, load_vad())
    clip_spans = merge_spans(spans, min_s=args.min_s, max_s=args.max_s)
    print(f"{len(spans)} speech spans → {len(clip_spans)} clips (3–11 s). Transcribing with "
          f"WhisperX {args.model} on {args.device}…")

    model = load_transcriber(args.model, args.device)

    def transcribe_fn(clip, csr):
        c16 = resample_audio(clip, csr, 16000) if csr != 16000 else clip
        t = transcribe_clip(c16.astype(np.float32), 16000, model)
        return t.text, t.confidence

    records, flagged = build_records(
        clip_spans, audio, sr, args.raw_dir, transcribe_fn=transcribe_fn,
        snr_gate=args.snr_gate, min_s=args.min_s, max_s=args.max_s)

    write_manifest(args.manifest, records)
    csv = args.manifest.rsplit(".", 1)[0] + "_transcripts.csv"
    write_transcript_csv(csv, records)
    print("\n" + summarize(records, flagged, snr_gate=args.snr_gate))
    print(f"\nmanifest → {args.manifest}\ntranscripts → {csv}")
    if flagged:
        print(f"\n{len(flagged)} flagged (review/re-record); run voxclone-verify to QC transcripts.")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
