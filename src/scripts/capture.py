"""voxclone-capture — guided dataset recorder (ALSA arecord, no PortAudio needed).

Walks the prompt bank one sentence at a time, records each from the mic with
push-to-stop, gates it on the prep rules (3-11 s, SNR >= 30 dB, low clipping), auto-names
clips ``<category>_<index>.wav`` under --raw-dir, and writes a prep-ready manifest pairing
each clip with its known transcript (no Whisper). Resumable: rerun to continue where you
left off; stops at the per-session minute cap to avoid fatigue drift.

    voxclone-capture --device plughw:1,0 --raw-dir experiments/danil/raw \\
                     --manifest experiments/danil/manifest.jsonl

Then: voxclone-prep / prep.split over the manifest.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from voxclone.capture.recorder import CALIBRATION_SENTENCE, SessionTotals, clip_path, session_cap_reached
from voxclone.capture.session import (
    CAPTURE_ORDER,
    accept_clip,
    arecord_cmd,
    load_records,
    ordered_prompts,
    remaining_prompts,
    save_records,
    to_record,
    upsert,
)

DEFAULT_DEVICE = "plughw:1,0"  # ALSA card 1 = the fifine USB mic on this box


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voxclone-capture", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=DEFAULT_DEVICE,
                   help="ALSA capture device (default plughw:1,0; see --list-devices)")
    p.add_argument("--raw-dir", default="experiments/danil/raw", help="where clips are written")
    p.add_argument("--manifest", default="experiments/danil/manifest.jsonl",
                   help="prep-ready manifest (JSONL of ClipRecord)")
    p.add_argument("--sr", type=int, default=48000, help="capture sample rate (downsampled in prep)")
    p.add_argument("--session-minutes", type=float, default=20.0,
                   help="stop after this many minutes of accepted audio (anti-fatigue cap)")
    p.add_argument("--order", nargs="+", default=list(CAPTURE_ORDER),
                   help="category order to record in")
    p.add_argument("--list-devices", action="store_true", help="list ALSA capture devices and exit")
    return p


def record_push_to_stop(device, sr, out_path, *, _input=input,
                        _popen=subprocess.Popen):  # pragma: no cover (hardware/interactive)
    """Start arecord, let the user read the sentence, stop on the next keypress."""
    _input("    ▶ Enter to START…")
    proc = _popen(arecord_cmd(device, sr, out_path))
    _input("    ⏹ recording… Enter to STOP.")
    proc.terminate()
    proc.wait()


def _fmt(ev) -> str:  # pragma: no cover (display)
    mark = "✓" if ev.ok else "✗"
    line = f"    {mark} {ev.duration:.1f}s  SNR {ev.snr_db:.0f}dB  peak {ev.peak:.2f}  clip {ev.clipped_fraction*100:.1f}%"
    return line if ev.ok else line + "  — " + "; ".join(ev.reasons)


def run(args) -> int:  # pragma: no cover (interactive/hardware)
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    records = load_records(args.manifest)
    order = tuple(args.order)
    remaining = remaining_prompts(raw, order)
    total = len(ordered_prompts(order))
    print(f"\nvoxclone-capture — {total - len(remaining)}/{total} already recorded, "
          f"{len(remaining)} to go. Device {args.device}.")
    print("How: read each sentence naturally, one clip each, 3-11 s, constant mic distance.")
    print(f"\nCalibration (read once to warm up / check level):\n  “{CALIBRATION_SENTENCE}”\n")

    totals = SessionTotals()
    for n, p in enumerate(remaining, 1):
        if session_cap_reached(totals, args.session_minutes):
            print(f"\n⏸  Session cap ({args.session_minutes:g} min) reached — rest your voice; "
                  f"rerun to continue.")
            break
        print(f"\n[{n}/{len(remaining)}]  {p.category} #{p.index}")
        print(f"    “{p.text}”")
        cmd = input("  [Enter]=record  s=skip  q=quit: ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            continue
        while True:  # re-take loop
            record_push_to_stop(args.device, args.sr, clip_path(raw, p.category, p.index))
            ev, rec = accept_clip(p, raw)
            print(_fmt(ev))
            if ev.ok:
                choice = input("    [Enter]=keep  r=retake  q=quit: ").strip().lower()
                if choice == "r":
                    continue
                if choice == "q":
                    p = None
                    break
                records = upsert(records, rec)
                save_records(args.manifest, records)
                totals.add(ev.duration)
                break
            choice = input("    [r]=retake (recommended)  k=keep anyway  s=skip: ").strip().lower()
            if choice == "k":
                records = upsert(records, to_record(p, clip_path(raw, p.category, p.index), ev))
                save_records(args.manifest, records)
                totals.add(ev.duration)
                break
            if choice == "s":
                break
            # default: retake
        if p is None:  # user quit mid-clip
            break

    save_records(args.manifest, records)
    print(f"\n✓ Saved {len(records)} clips → {args.manifest} "
          f"(+{totals.total_minutes:.1f} min this session). Next: voxclone-prep over the manifest.")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:  # pragma: no cover (hardware)
        subprocess.run(["arecord", "-l"])
        return 0
    return run(args)  # pragma: no cover (interactive)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
