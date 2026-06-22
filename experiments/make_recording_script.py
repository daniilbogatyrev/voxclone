"""Render the voxclone capture prompt bank into a human-readable, session-organized
read-off script for the dataset recording. Derived from voxclone.capture.prompts (the
source of truth) so it never drifts. Run with the project venv:

    /home/prada/code/danill/.venv/bin/python experiments/make_recording_script.py
"""
from pathlib import Path

from voxclone.capture.prompts import PROMPTS
from voxclone.capture.recorder import CALIBRATION_SENTENCE

OUT = Path(__file__).parent / "danil" / "recording_script.md"
WPM = 150  # conversational read-aloud
SESSION_MIN = 18  # keep each session under the 20-min anti-fatigue cap


def mins(words: int) -> float:
    return words / WPM


def words(sentences: list[str]) -> int:
    return sum(len(s.split()) for s in sentences)


# Style prompts first (highest prosodic value), then Harvard for phonetic coverage.
ORDER = ["expressive", "conversational", "technical", "harvard"]
flat: list[tuple[str, str]] = [(cat, s) for cat in ORDER for s in PROMPTS[cat]]

# Pack into sessions of ~SESSION_MIN of estimated speech.
sessions: list[list[tuple[str, str]]] = [[]]
acc = 0.0
for cat, s in flat:
    w = len(s.split())
    if mins(acc + w) > SESSION_MIN and sessions[-1]:
        sessions.append([])
        acc = 0.0
    sessions[-1].append((cat, s))
    acc += w

lines: list[str] = []
add = lines.append
add("# Voice-clone recording script — danil (English)\n")
add(f"**{len(flat)} sentences · ~{mins(words([s for _, s in flat])):.0f} min of speech · "
    f"{len(sessions)} sessions (~{SESSION_MIN} min each).**\n")
add("## How to record (read this once)\n")
for tip in [
    "Quiet room, no fan/AC hum. Aim for **SNR ≥ 30 dB** (the prep step will reject noisy clips).",
    "Keep a **constant mic distance** (~a hand-span) and the **same room/mic** for every session.",
    "**One sentence = one clip.** Read it naturally, then pause ~1 s before the next.",
    "Keep clips **3–11 seconds**. If you stumble, just re-read that sentence (re-record the clip).",
    "Read in your **natural** voice — not a 'radio' voice. Match the punctuation (a `?` rises, a `!` lifts).",
    "Take breaks. **Stop at ~18–20 min per session** to avoid your voice drifting when tired.",
    "Save clips as mono WAV. Naming like `harvard_0001.wav` is ideal (the prep step sorts by category).",
]:
    add(f"- {tip}")
add("\n### Calibration — read this at the **start of every session** (sets your levels):\n")
add(f"> {CALIBRATION_SENTENCE}\n")
add("---\n")

n = 0
for si, sess in enumerate(sessions, 1):
    sw = words([s for _, s in sess])
    add(f"## Session {si} — {len(sess)} sentences (~{mins(sw):.0f} min)\n")
    add(f"_Read the calibration sentence first. Then:_\n")
    cur = None
    for cat, s in sess:
        if cat != cur:
            cur = cat
            add(f"\n**{cat.capitalize()}**\n")
        n += 1
        add(f"{n}. {s}")
    add("")
    add("_— break / stop here —_\n")
    add("---\n")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT} ({len(flat)} sentences, {len(sessions)} sessions)")
