"""Render the capture prompt bank into a clean, paginated PDF read-off script.

Derived from voxclone.capture.prompts (source of truth), same session layout as
make_recording_script.py. Uses DejaVuSans so em-dashes / smart quotes render correctly.

    /home/prada/code/danill/.venv/bin/python experiments/make_recording_pdf.py
"""
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from voxclone.capture.prompts import PROMPTS
from voxclone.capture.recorder import CALIBRATION_SENTENCE

OUT = Path(__file__).parent / "danil" / "recording_script.pdf"
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
WPM = 150
SESSION_MIN = 18
ORDER = ["expressive", "conversational", "technical", "harvard"]


def words(seq):
    return sum(len(s.split()) for s in seq)


def mins(w):
    return w / WPM


# Pack prompts into ~SESSION_MIN sessions (style prompts first, then Harvard).
flat = [(c, s) for c in ORDER for s in PROMPTS[c]]
sessions, acc = [[]], 0
for c, s in flat:
    w = len(s.split())
    if mins(acc + w) > SESSION_MIN and sessions[-1]:
        sessions.append([]); acc = 0
    sessions[-1].append((c, s)); acc += w

pdf = FPDF(format="A4")
pdf.add_font("DejaVu", "", f"{FONT_DIR}/DejaVuSans.ttf")
pdf.add_font("DejaVu", "B", f"{FONT_DIR}/DejaVuSans-Bold.ttf")
pdf.add_font("DejaVu", "I", f"{FONT_DIR}/DejaVuSans-Oblique.ttf")
pdf.set_auto_page_break(auto=True, margin=16)


def line(txt, size=11, style="", lh=6.0, fill=False):
    pdf.set_font("DejaVu", style, size)
    pdf.multi_cell(0, lh, txt, fill=fill, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def gap(n=2):
    pdf.ln(n)


# ---- cover / how-to ----
pdf.add_page()
line("Voice-clone recording script — danil (English)", 18, "B", 10)
line(f"{len(flat)} sentences · ~{mins(words([s for _, s in flat])):.0f} min · "
     f"{len(sessions)} sessions (~{SESSION_MIN} min each)", 11, "I", 6)

line("How to record", 13, "B", 8)
for tip in [
    "Quiet room, no fan/AC hum. Aim for SNR ≥ 30 dB (the prep step rejects noisy clips).",
    "Constant mic distance (~a hand-span) and the same room/mic every session.",
    "One sentence = one clip. Read it naturally, pause ~1 s, then the next.",
    "Keep clips 3–11 seconds. Stumbled? Just re-read that sentence.",
    "Natural voice — not a 'radio' voice. Match the punctuation (a ? rises, a ! lifts).",
    "Take breaks. Stop at ~18–20 min per session to avoid voice drift.",
    "Export mono WAV (not mp3 — lossy hurts the clone).",
]:
    line(f"•  {tip}", 11, "", 6)
gap(3)
line("Calibration — read at the start of every session:", 12, "B", 4)
pdf.set_fill_color(238, 238, 238)
line(f"  {CALIBRATION_SENTENCE}", 12, "I", 8, fill=True)

# ---- sessions ----
n = 0
for si, sess in enumerate(sessions, 1):
    pdf.add_page()
    line(f"Session {si}", 16, "B", 6)
    line(f"{len(sess)} sentences · ~{mins(words([s for _, s in sess])):.0f} min  —  "
         f"read the calibration line first.", 10, "I", 4)
    cur = None
    for c, s in sess:
        if c != cur:
            cur = c
            gap(1)
            line(c.capitalize(), 12, "B", 7)
        n += 1
        line(f"{n}.  {s}", 11, "", 6.4)

OUT.parent.mkdir(parents=True, exist_ok=True)
pdf.output(str(OUT))
print(f"wrote {OUT} — {OUT.stat().st_size // 1024} KB, {len(flat)} sentences, "
      f"{len(sessions)} sessions, {pdf.pages_count} pages")
