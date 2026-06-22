"""Builder for ``notebooks/voice_cloning_study.ipynb`` — the project's one notebook.

A single, self-contained notebook that guides **a reader through cloning their OWN
voice** and documents the project's results. It (1) walks through the training
pipeline, (2) documents & compares the real results (loaded from the committed
``experiments/ab_eval_finetune_results.json`` and re-scored with the REAL
``voxclone.eval.picker.combined_score`` — not a paraphrase), (3) explains exactly how
the "≈90 % of the real voice" number is computed, (4) lets the reader **clone their
own voice zero-shot** (few-shot, from one uploaded clip) and (5) documents how to
**fine-tune** their own voice end-to-end (links the read-aloud script + the runbook).

Two audiences from one source:

* ``public`` (default) — the version committed to the public GitHub repo. It contains
  **no audio of the author and no fine-tuned weights**; the author's measured numbers
  (bar charts / tables only) stay. The "hear the author's voice" section is omitted.
* ``professor`` — the version shipped (out of band) in the starter-kit ZIP alongside
  the author's private fine-tuned weights. It keeps the "hear the author's cloned
  voice" section so it can be executed once on the GPU box to embed audio.

Prose is German (the examiner / fellow-student context); code + identifiers stay English.

Build (GPU-free) from the repo root:

    .venv/bin/python notebooks/build_voice_cloning_study.py                # public
    .venv/bin/python notebooks/build_voice_cloning_study.py --audience professor

Optionally pre-embed the GPU-free analysis outputs while leaving the live audio cells
dormant — set the force-no-gen flag so the generation cells print a neutral note
instead of launching a worker:

    VOXCLONE_NB_FORCE_NOGEN=1 .venv/bin/python -m nbconvert --to notebook --execute --inplace \
        notebooks/voice_cloning_study.ipynb \
        --ExecutePreprocessor.kernel_name=voxclone --ExecutePreprocessor.timeout=240
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

AUDIENCES = ("public", "professor")


def _build_cells(audience: str = "public") -> list:
    """Assemble the notebook cells for ``audience`` ('public' | 'professor')."""
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}, got {audience!r}")
    is_prof = audience == "professor"

    cells: list = []
    def md(s): cells.append(new_markdown_cell(s.strip("\n")))
    def code(s): cells.append(new_code_cell(s.strip("\n")))

    # ----------------------------------------------------------------------- #
    # 0 · Titel + Bedienung                                                    #
    # ----------------------------------------------------------------------- #
    hear_line = (
        "5. **Meine Stimme hören** — meine geklonte Stimme, live im Notebook.\n"
        if is_prof else ""
    )
    md(rf"""
# Voice Cloning — *deine* Stimme klonen und *messen*, ob es funktioniert

**Daniil Bogatyrev · Data Science & KI, DHBW Lörrach · persönlich, nicht-kommerziell**

Dieses Notebook ist das Herzstück des Projekts und zugleich eine **Anleitung: klone deine
eigene Stimme**. Es vergleicht vier quelloffene TTS-Engines — **F5-TTS, XTTS-v2,
Chatterbox, GPT-SoVITS** — jeweils *zero-shot* (ohne Training) und, wo sinnvoll,
*fine-getunt* (mit Training), gemessen an **einer einwilligenden Stimme** (meiner, mit
Einwilligung für dieses Projekt aufgenommen).

Es führt dich durch:

1. **Den Trainingsprozess** — wie aus einer Aufnahme ein Klon wird (Pipeline).
2. **Die Ergebnisse** — die Rangliste, jede Zahl aus echtem Code neu hergeleitet, plus Diagramme.
3. **Die Schlüsselzahl** — wie genau „F5 bildet die Stimme zu ≈ 90 % ab" berechnet wird.
{hear_line}4. **Few-Shot — *deine* Stimme klonen** — ein kurzer Clip von dir, sofort geklont (kein Training).
5. **Fine-Tuning — *deine* Stimme trainieren** — der komplette Weg von der Aufnahme zum eigenen Modell.

> **Bedienung:** Kernel **`VoxClone (.venv)`** wählen (oben rechts), dann **Run All**.
> Die Analyse-Abschnitte laufen **GPU-frei** und sofort. Die Audio-Abschnitte brauchen
> die GPU-Box mit vollständigem Setup; der erste Clip pro Stimme wärmt die GPU auf
> (~60–90 s), danach geht es schnell. **Du brauchst nie ein Terminal** — das Notebook
> startet alles selbst.

*Ethik: Klone nur Stimmen, für die du die ausdrückliche Einwilligung der sprechenden Person hast.*
""")

    # ----------------------------------------------------------------------- #
    # 0 · Pre-flight                                                           #
    # ----------------------------------------------------------------------- #
    md(r"""
## 0 · Vorbereitung (Pre-flight)

Prüft den Kernel und meldet, was auf dieser Maschine vorhanden ist. Die Analyse-Teile laufen
immer; ob die Audio-Teile laufen, hängt davon ab, ob die Conda-Envs + Modelle da sind
(`CAN_GENERATE`).
""")
    code(r"""
import os, sys

# Capability check: the right kernel is the one where 'voxclone' imports (stdlib-thin .venv).
try:
    from voxclone.clone import studio
except ImportError as e:
    raise SystemExit(
        "Falscher Kernel — dieses Notebook braucht den Projekt-Kernel 'VoxClone (.venv)'.\n"
        "  Kernel-Menü (oben rechts) → 'VoxClone (.venv)', dann Run All.\n"
        f"  Aktueller Kernel: {sys.executable}\n  ({e})"
    )

missing_assets = studio.missing_assets()
missing_envs   = studio.missing_envs()
# Validation hook: lets a head-less build embed the analysis outputs without launching a GPU worker.
FORCE_NOGEN  = os.environ.get("VOXCLONE_NB_FORCE_NOGEN") == "1"
CAN_GENERATE = (not missing_assets) and (not missing_envs) and (not FORCE_NOGEN)

print("Kernel :", sys.executable)
print("Engines:", ", ".join(v.key for v in studio.VOICES))
if CAN_GENERATE:
    print("\n✅ Alles bereit — die Audio-Abschnitte (Stimme klonen) funktionieren.")
else:
    print("\nℹ️  Die Analyse-Abschnitte laufen trotzdem (GPU-frei).")
    if missing_envs:   print("   Für Audio fehlen Conda-Envs   :", ", ".join(missing_envs))
    if missing_assets: print("   Für Audio fehlen Modelle/Assets:", len(missing_assets), "Datei(en)")
    print("   → Für die Audio-Abschnitte auf der GPU-Box mit vollständigem Setup ausführen.")
""")

    # ----------------------------------------------------------------------- #
    # 1 · Die Frage                                                            #
    # ----------------------------------------------------------------------- #
    md(r"""
## 1 · Die Frage

Ich lerne gern Sprachen und wollte sie **in meiner eigenen Stimme** hören. Daraus wurde eine
messbare Frage:

> **Welche quelloffene TTS-Engine klont eine Stimme am besten — und lohnt sich Fine-Tuning?**

Die Kandidaten und Familien:

| Engine | Familie | Modus |
|---|---|---|
| **F5-TTS** | Flow-Matching-Transformer | zero-shot **und** fine-getunt |
| **XTTS-v2** | Autoregressiv (GPT-Stil) | zero-shot **und** fine-getunt |
| **Chatterbox** | Llama-Backbone-LM | nur zero-shot (Projektentscheidung) |
| **GPT-SoVITS v2Pro** | Autoregressiv + VITS | nur zero-shot (Vergleich) |

**Zero-shot** = die Stimme wird zur Laufzeit aus einem ~9-Sekunden-Referenzclip kopiert, ohne
Training. **Fine-getunt** = das Modell wird auf einem Datensatz der Stimme weitertrainiert.
""")

    # ----------------------------------------------------------------------- #
    # 2 · Der Trainingsprozess                                                 #
    # ----------------------------------------------------------------------- #
    md(r"""
## 2 · Der Trainingsprozess — von der Aufnahme zum Klon

Eine durchgehende Pipeline, jede Stufe ist echter Code im Paket `voxclone`:

```
 Aufnehmen  →  Aufbereiten  →  Splitten      →  Fine-Tuning      →  Bewerten
 (capture)     (prep)          (split)           (train)             (eval)
 ~34 min       entrauschen,    184 train /       je Engine in        Score je
 eine Stimme   VAD-schneiden,  20 held-out /     eigener Conda-Env   Kandidat
               24 kHz, Whisper Referenzclip      (GPU-Batch-Job)
```

**Aufnahme.** Eine einzige ~33,6-minütige Aufnahme, geleitet von Lese-Prompts aus
`voxclone.capture.prompts` (Harvard-Sätze + expressive / konversationelle / technische Sätze).
Anforderungen, die der Prep-Validator erzwingt: **24 kHz mono, SNR ≥ 30 dB**, Clips ~3–11 s.
Das vollständige Lese-Skript (814 Sätze) liegt in
[`data/recording_script.md`](../data/recording_script.md) — daran führt dich Abschnitt 5 entlang.

**Aufbereitung (`voxclone.prep`).** Resampling → Entrauschen → VAD-Segmentierung →
Peak-Normalisierung → Transkription (Whisper `large-v3`) → Validierung. Ergebnis: ein Manifest
aus **216 sauberen Clips**, daraus **184 Trainings-** + **20 Held-out-Clips** und ein
**9,0-Sekunden-Referenzclip**.

**Fine-Tuning (`voxclone.train`, je Engine in eigener Conda-Env).** XTTS-v2: 15 Epochen,
lr 5e-6. F5-TTS: 100 Epochen, lr 1e-5. *(Das Training selbst ist ein GPU-Batch-Job und läuft
nicht live in diesem Notebook — dieses Notebook dokumentiert ihn und arbeitet mit seinen
festgeschriebenen Ergebnissen.)* GPT-SoVITS bleibt Vergleich (zero-shot), Chatterbox per
Projektentscheidung nur zero-shot.

**Bewertung (`voxclone.eval`).** Jeder Kandidat erhält **einen** zusammengesetzten Score auf
denselben Held-out-Sätzen — Details unten. Die Zelle zeigt, dass Prompts, Gewichte und
Datensatz-Fakten aus echtem Code/Daten stammen:
""")
    code(r"""
from voxclone.capture import prompts
from voxclone.common.config import EvalConfig
import json, pathlib, voxclone

ROOT = pathlib.Path(voxclone.__file__).resolve().parents[2]

# (a) the real recording prompt bank
print("Aufnahme-Prompts (Kategorie: Anzahl):",
      {k: len(v) for k, v in prompts.PROMPTS.items()})

# (b) the real scoring weights live in code, not in a slide
w = EvalConfig().weights()
print("Score-Gewichte (EvalConfig):", w,
      "| WER-Disqualifikation > ", EvalConfig().wer_dq_threshold)

# (c) dataset / eval facts straight from the committed results file
res = json.loads((ROOT / "experiments" / "ab_eval_finetune_results.json").read_text())
print("\nBewertungslauf      :", res["date"])
print("Sätze je Bedingung  :", res["n_sentences"], "(held-out, English)")
print("Echtstimmen-UTMOS   :", round(res["real_utmos_heldout"], 3))
print("Obergrenze (real↔real):", round(res["ceiling"], 4))
""")

    # ----------------------------------------------------------------------- #
    # 3 · Ergebnisse                                                           #
    # ----------------------------------------------------------------------- #
    md(r"""
## 3 · Ergebnisse & Vergleich

Der Score je Bedingung ist **eine** Formel (`voxclone.eval.picker.combined_score`):

$$\text{Score} \;=\; 0{,}30 \cdot \frac{\text{Ähnlichkeit}}{\text{Obergrenze}} \;+\; 0{,}50 \cdot \frac{\text{UTMOS}}{5} \;+\; 0{,}20 \cdot (1 - \text{WER})$$

| Achse | Gewicht | Was | Wie |
|---|---|---|---|
| **Ähnlichkeit** | 0,30 | gleiche Stimme? | ECAPA-Sprecher-Embedding, Kosinus zum Centroid echter Clips |
| **Natürlichkeit** | 0,50 | klingt menschlich? | UTMOS (neuronaler MOS-Schätzer, 1–5) |
| **Klarheit (WER)** | 0,20 | verständlich? | WhisperX `large-v3`, Wortfehlerrate |

Die Rangliste, direkt aus den festgeschriebenen Ergebnissen:
""")
    code(r"""
import pandas as pd

df = (pd.DataFrame(res["rows"])
        .sort_values("score", ascending=False)
        .reset_index(drop=True))
df.index = df.index + 1                      # 1-based rank
view = df[["engine", "mode", "similarity", "pct_ceiling", "naturalness", "wer", "score"]].copy()
view["pct_ceiling"] = (view["pct_ceiling"] * 100).round(1).astype(str) + " %"
view.round({"similarity": 4, "naturalness": 3, "wer": 4, "score": 4})
""")

    md(r"""
### Der Score ist echter Code, keine Beschreibung

Damit nichts „nur behauptet" ist: Wir schicken die **gespeicherten** Roh-Metriken jeder Zeile
durch die **echte** Scoring-Funktion und vergleichen mit dem **gespeicherten** Score. Die
Differenz ist ~ 0 → die Rangliste *ist* exakt diese Formel auf den gemessenen Werten.
""")
    code(r"""
import inspect
from voxclone.eval.picker import combined_score

w, ceiling = res["weights"], res["ceiling"]
chk = pd.DataFrame([{
    "engine": r["engine"], "mode": r["mode"],
    "gespeichert": r["score"],
    "neu_berechnet": combined_score(r["similarity"], r["naturalness"], r["wer"], w, ceiling=ceiling),
} for r in res["rows"]])
chk["|Δ|"] = (chk["neu_berechnet"] - chk["gespeichert"]).abs()
print("max |gespeichert − neu_berechnet| über alle 6 Zeilen:", chk["|Δ|"].max())
print("→ die veröffentlichten Scores sind exakt diese Formel auf den gemessenen Werten.\n")
print(inspect.getsource(combined_score))
chk.sort_values("gespeichert", ascending=False).reset_index(drop=True).round(10)
""")

    md(r"""
### Die Diagramme (jedes belegt eine Aussage)
""")
    code(r"""
import matplotlib.pyplot as plt

# dark palette echoing the deck / studio for a cohesive look
BG, FG, ACCENT, MUTED, ACC2 = "#11131a", "#e6e8ef", "#fa520f", "#8b90a3", "#ffa110"

def dark(ax, fig):
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for s in ax.spines.values(): s.set_color(MUTED)
    ax.tick_params(colors=FG); ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)

rows_asc = sorted(res["rows"], key=lambda r: r["score"])     # ascending → best on top of barh
labels   = [f'{r["engine"]} · {r["mode"]}' for r in rows_asc]
is_f5    = [r["engine"] == "F5-TTS" for r in rows_asc]
""")
    code(r"""
# Diagramm 1 — Rangliste (zusammengesetzter Score). F5 besetzt die obersten zwei Plätze.
fig, ax = plt.subplots(figsize=(8, 3.4))
scores = [r["score"] for r in rows_asc]
ax.barh(labels, scores, color=[ACCENT if f else MUTED for f in is_f5])
ax.set_xlim(0.70, 0.90)
ax.set_xlabel("Score = 0,30·Ähnl. + 0,50·UTMOS/5 + 0,20·(1−WER)   (Achse beschnitten)")
ax.set_title("Rangliste — F5 führt in beiden Modi")
for i, v in enumerate(scores):
    ax.text(v + 0.002, i, f"{v:.4f}", color=FG, va="center", fontsize=9)
dark(ax, fig); fig.tight_layout(); plt.show()
""")
    code(r"""
# Diagramm 2 — Ähnlichkeit als % der Echtstimmen-Obergrenze (der eigentliche Trenner).
fig, ax = plt.subplots(figsize=(8, 3.4))
pct = [r["pct_ceiling"] * 100 for r in rows_asc]
ax.barh(labels, pct, color=[ACCENT if f else MUTED for f in is_f5])
ax.axvline(100, ls="--", color=FG, label="echte Stimme (Obergrenze)")
ax.set_xlim(0, 112)
ax.set_xlabel("Sprecherähnlichkeit, % der Echtstimmen-Obergrenze")
ax.set_title("Ähnlichkeit trennt die Engines (F5 ≈ 90 %, GPT-SoVITS ≈ 53 %)")
for i, v in enumerate(pct):
    ax.text(v + 1, i, f"{v:.0f}%", color=FG, va="center", fontsize=9)
ax.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=8)
dark(ax, fig); fig.tight_layout(); plt.show()
""")
    code(r"""
# Diagramm 3 — Fine-Tuning-Steigung: der Nutzen hängt komplett von der Engine ab.
by = {(r["engine"], r["mode"]): r["pct_ceiling"] * 100 for r in res["rows"]}
fig, ax = plt.subplots(figsize=(6, 4))
for eng, color in [("F5-TTS", ACCENT), ("XTTS-v2", ACC2)]:
    ys = [by[(eng, "zero-shot")], by[(eng, "fine-tuned")]]
    ax.plot([0, 1], ys, "-o", color=color, label=eng, linewidth=2)
    for x, y in zip([0, 1], ys):
        ax.text(x, y + 1.2, f"{y:.0f}%", color=color, ha="center", fontsize=9)
ax.axhline(by[("F5-TTS", "zero-shot")], ls=":", color=MUTED)
ax.set_xticks([0, 1]); ax.set_xticklabels(["zero-shot", "fine-getunt"])
ax.set_ylabel("% der Obergrenze"); ax.set_ylim(50, 100)
ax.set_title("Fine-Tuning-Effekt hängt von der Engine ab\n(F5 zero-shot schlägt XTTS fine-getunt)")
ax.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=9)
dark(ax, fig); fig.tight_layout(); plt.show()
""")
    code(r"""
# Diagramm 4 — Natürlichkeit (UTMOS) mit der Linie der echten Aufnahme. Sie ist gesättigt.
order = sorted(res["rows"], key=lambda r: -r["score"])
names = [f'{r["engine"]}\n{r["mode"]}' for r in order]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(names, [r["naturalness"] for r in order],
       yerr=[r.get("utmos_std", 0) for r in order], color=ACCENT, ecolor=MUTED, capsize=3)
ax.axhline(res["real_utmos_heldout"], ls="--", color=FG,
           label=f'echte Stimme ({res["real_utmos_heldout"]:.2f})')
ax.set_ylim(1, 5); ax.set_ylabel("UTMOS (1–5)")
ax.set_title("Natürlichkeit ist gesättigt — jeder Klon liegt über der echten Aufnahme")
ax.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=8)
dark(ax, fig); fig.tight_layout(); plt.show()
""")

    md(r"""
**Was die Diagramme zeigen.** (1) Es gibt eine klare Rangliste — F5 besetzt beide Spitzenplätze.
(2) **Ähnlichkeit** ist der Trenner (Natürlichkeit ist gesättigt, WER nahe null), also ist es im
Kern ein Ähnlichkeits-Wettbewerb. (3) Der Fine-Tuning-Nutzen hängt von der Engine ab: XTTS steigt
stark (+11 Punkte), F5 ist schon oben (+1,3) — und **F5 ohne Training schlägt XTTS voll
fine-getunt**. Die eigentliche Lehre: *die richtige Engine zu wählen schlägt das Fine-Tuning der
falschen.*
""")

    # ----------------------------------------------------------------------- #
    # 4 · Wie die 90 % entstehen                                               #
    # ----------------------------------------------------------------------- #
    md(r"""
## 4 · Wie die „≈ 90 %" entstehen (die Schlüsselzahl)

Die wichtigste Zahl des Projekts: **F5 bildet die Stimme zu ≈ 90 % ab.** So wird sie berechnet —
Schritt für Schritt, mit dem echten Code:

1. **Stimme → Vektor.** ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) verwandelt jeden Clip in
   einen „Stimm-Fingerabdruck" — einen Vektor, der kodiert *wer* spricht, nicht *was*.
2. **Ähnlichkeit = Kosinus.** Kosinus-Ähnlichkeit zwischen dem Fingerabdruck des Klons und dem
   Mittelwert (Centroid) der echten Clips. F5 fine-getunt roh: **0,7466**.
3. **Die ehrliche Obergrenze.** Selbst **zwei echte Aufnahmen derselben Person** erreichen keine
   1,0 (andere Sätze, Betonung, Mikro). Der mittlere paarweise Kosinus der echten Clips ist die
   Obergrenze: **0,8284** — die ehrlichen „100 %".
4. **Als Prozent:** $0{,}7466 / 0{,}8284 = \mathbf{90{,}1\,\%}$ (F5 zero-shot: 88,8 %).

Also heißt „90 %": F5 kommt der Stimme **so nah, wie zwei echte Aufnahmen derselben Person sich
gegenseitig kommen** — *nicht* 90 % einer perfekten 1,0. Die nächste Zelle zeigt den echten
Mess-Code (inkl. der Kosinus-Funktion selbst) und rechnet die Prozent aus den gespeicherten
Zahlen nach:
""")
    code(r"""
from voxclone.eval import similarity

# the real measurement code (no paraphrase): cosine, the real-vs-real ceiling, the score
print(inspect.getsource(similarity.cosine))
print(inspect.getsource(similarity.reference_ceiling))
print(inspect.getsource(similarity.similarity_score))

f5_ft = next(r for r in res["rows"] if r["engine"] == "F5-TTS" and r["mode"] == "fine-tuned")
pct = f5_ft["similarity"] / res["ceiling"] * 100
print(f'F5 (fine-getunt) Roh-Ähnlichkeit : {f5_ft["similarity"]:.4f}')
print(f'Obergrenze (real↔real)           : {res["ceiling"]:.4f}')
print(f'→ % der Obergrenze               : {pct:.1f}%   (auf der Folie als ≈ 90 %)')
""")

    # ----------------------------------------------------------------------- #
    # 5 · Daniils Stimme hören  (PROFESSOR-Build only)                         #
    # ----------------------------------------------------------------------- #
    if is_prof:
        md(r"""
## 5 · Hör es selbst — meine geklonte Stimme

*(Braucht die GPU-Box mit vollständigem Setup und die privaten fine-getunten Gewichte.)*
Bearbeite `TEXT` / `VOICE` / `LANG` und führe die Zelle aus. Beim **Run All** spielt dies
automatisch mit der besten Stimme. Der **erste** Clip ist langsam (GPU-Aufwärmen ~60–90 s),
danach geht es schnell.
""")
        code(r"""
TEXT  = "Hi, this is my cloned voice, generated live inside this notebook."
VOICE = "f5_finetuned"   # f5_finetuned · f5_zeroshot · xtts_finetuned · xtts_zeroshot · chatterbox
LANG  = "en"             # "en", oder "de" (jede Stimme außer f5_finetuned)

from IPython.display import Audio, display
if not CAN_GENERATE:
    print("ℹ️  Diese Zelle erzeugt live Audio — auf der GPU-Box ausführen "
          "(Kernel: VoxClone (.venv), Run All).")
else:
    print(f"Wärme '{VOICE}' auf der GPU auf — der ERSTE Clip dauert ~60–90 s, danach schnell …")
    try:
        wav, sr, path = studio.say(VOICE, TEXT, language=LANG)
        display(Audio(wav, rate=sr))
    except Exception as e:
        print("Aufwärmen fehlgeschlagen — Kernel > Restart and Run All, oder Panel unten nutzen.")
        print("Details:", e)
""")
        md(r"""
**Interaktiv:** Sprache + Stimme wählen, Satz tippen, **Generate**. Deutsch ist *cross-lingual*
(meine Stimme spricht Deutsch aus einer englischen Referenz); **F5 fine-getunt ist Englisch-only**
und fällt bei Deutsch aus der Liste. Stimme wechseln lädt einmal neu (~60–90 s).
""")
        code(r"""
try:
    import ipywidgets as W
    from IPython.display import Audio, display

    def _voice_opts(lang):
        return [(v.label, v.key) for v in studio.voices_for(lang)]

    lang_dd  = W.Dropdown(options=[("English", "en"), ("Deutsch (German)", "de")],
                          value="en", description="Sprache:", layout=W.Layout(width="320px"))
    voice_dd = W.Dropdown(options=_voice_opts("en"), value=studio.DEFAULT_VOICE,
                          description="Stimme:", layout=W.Layout(width="460px"))
    text_in  = W.Textarea(value="Tippe hier einen beliebigen Satz und klicke Generate.",
                          description="Sag:", layout=W.Layout(width="680px", height="70px"))
    gen_btn  = W.Button(description="Generate", button_style="primary", icon="play")
    rst_btn  = W.Button(description="Restart voice", icon="refresh")
    out      = W.Output()

    def _on_lang(change):
        voice_dd.options = _voice_opts(change["new"])
        voice_dd.value = studio.DEFAULT_GERMAN_VOICE if change["new"] == "de" else studio.DEFAULT_VOICE
    lang_dd.observe(_on_lang, names="value")

    def _generate(_):
        with out:
            out.clear_output()
            if not CAN_GENERATE:
                print("ℹ️  Live-Audio nur auf der GPU-Box mit vollständigem Setup."); return
            try:
                wav, sr, path = studio.say(voice_dd.value, text_in.value, language=lang_dd.value)
                display(Audio(wav, rate=sr))
            except Exception as e:
                print("Generierung fehlgeschlagen:", e)
                print('Tipp: "Restart voice", dann erneut Generate.')

    def _restart(_):
        with out:
            out.clear_output(); studio.restart(voice_dd.value)

    gen_btn.on_click(_generate); rst_btn.on_click(_restart)
    display(W.VBox([lang_dd, voice_dd, text_in, W.HBox([gen_btn, rst_btn]), out]))
except Exception as e:
    print("Interaktives Panel nicht verfügbar:", e)
    print("Nutze die Zelle oben — TEXT/VOICE/LANG bearbeiten und erneut ausführen.")
""")

    # ----------------------------------------------------------------------- #
    # Few-Shot · Klone deine eigene Stimme                                     #
    # ----------------------------------------------------------------------- #
    md(r"""
## {n} · Few-Shot — klone *deine eigene* Stimme (zero-shot, ohne Training)

Der schnellste Weg zu deiner geklonten Stimme — **kein Training, nur ein kurzer Clip**:

1. Nimm **~6–12 Sekunden** von dir auf (ruhig, sauber, ein Sprecher) und speichere als **mono WAV**.
   Lies dafür am besten genau diesen **Few-Shot-Satz** (der Kalibriersatz aus dem Lese-Skript) —
   er ist phonetisch ausgewogen und liefert eine gute Referenz:

   > *„This is my natural speaking voice, calm, clear, and steady, as I read these few lines aloud today."*

   Mehr Sätze zum Vorlesen findest du im vollständigen Lese-Skript:
   [`data/recording_script.md`](../data/recording_script.md) (814 Sätze).
2. **Lade den Clip hoch** und tippe, **was darin gesagt wird** (das Transkript — F5 nutzt es; XTTS/
   Chatterbox ignorieren es). Wenn du den Few-Shot-Satz oben gelesen hast, kopiere ihn einfach hier hinein.
3. Wähle eine **zero-shot-Engine** und tippe den Satz, den *deine* geklonte Stimme sagen soll.

Nur die **zero-shot**-Stimmen können eine fremde Referenz klonen. Bestes Ergebnis: **F5 (zero-shot)**.

> ⚠️ **Einwilligung:** Lade nur deine eigene Stimme hoch (oder eine, für die du ausdrücklich die
> Erlaubnis hast). Genau wie dieses Projekt: geklont wird nur mit Zustimmung.
""".format(n=6 if is_prof else 5))
    code(r"""
import tempfile, pathlib
try:
    import ipywidgets as W
    import soundfile as sf
    from IPython.display import Audio, display

    ref_up    = W.FileUpload(accept=".wav,.flac", multiple=False, description="Referenz-Clip")
    ref_txt   = W.Textarea(value="This is my natural speaking voice, calm, clear, and steady, "
                                 "as I read these few lines aloud today.",
                           placeholder="Was in deinem Clip gesagt wird (für F5)…",
                           description="Transkript:", layout=W.Layout(width="680px", height="56px"))
    say_txt   = W.Textarea(value="Hi, this is my own voice, cloned zero-shot in this notebook.",
                           description="Sag:", layout=W.Layout(width="680px", height="56px"))
    eng_dd    = W.Dropdown(options=[("F5 (zero-shot) — best", "f5_zeroshot"),
                                    ("XTTS (zero-shot)", "xtts_zeroshot"),
                                    ("Chatterbox", "chatterbox")],
                           value="f5_zeroshot", description="Engine:", layout=W.Layout(width="360px"))
    lang2_dd  = W.Dropdown(options=[("English", "en"), ("Deutsch (German)", "de")],
                           value="en", description="Sprache:", layout=W.Layout(width="320px"))
    clone_btn = W.Button(description="Clone my voice", button_style="primary", icon="play")
    out2      = W.Output()

    def _uploaded_bytes(upload):
        v = upload.value
        items = list(v.values()) if isinstance(v, dict) else list(v)   # v7 dict vs v8 tuple
        if not items:
            return None, None
        item = items[0]
        content = item["content"]
        data = content.tobytes() if hasattr(content, "tobytes") else bytes(content)
        return item.get("name", "upload.wav"), data

    def _clone(_):
        with out2:
            out2.clear_output()
            if not CAN_GENERATE:
                print("ℹ️  Live-Audio nur auf der GPU-Box mit vollständigem Setup."); return
            name, data = _uploaded_bytes(ref_up)
            if not data:
                print("Bitte zuerst einen mono-WAV-Clip (~6–12 s) hochladen."); return
            tmp = pathlib.Path(tempfile.gettempdir()) / "my_voice_ref.wav"
            tmp.write_bytes(data)
            try:
                info = sf.info(str(tmp))
                print(f"Referenz: {name} — {info.duration:.1f}s, {info.samplerate} Hz, "
                      f"{info.channels} Kanal/Kanäle")
                if info.duration < 3:
                    print("⚠️  sehr kurz — 6–12 s geben deutlich bessere Klone.")
            except Exception as e:
                print("Konnte die Datei nicht als Audio lesen — bitte eine echte WAV/FLAC hochladen.")
                print("Details:", e); return
            try:
                wav, sr, path = studio.say(eng_dd.value, say_txt.value, language=lang2_dd.value,
                                           ref_path=str(tmp), ref_text=(ref_txt.value or None))
                print("✓ deine geklonte Stimme:")
                display(Audio(wav, rate=sr))
            except Exception as e:
                print("Klonen fehlgeschlagen:", e)
                print('Tipp: zero-shot-Engine wählen; bei F5 das Transkript ausfüllen.')

    clone_btn.on_click(_clone)
    display(W.VBox([ref_up, ref_txt, eng_dd, lang2_dd, say_txt, clone_btn, out2]))
except Exception as e:
    print("Klon-Panel nicht verfügbar:", e)
""")

    # ----------------------------------------------------------------------- #
    # Fine-Tuning · Trainiere deine eigene Stimme                              #
    # ----------------------------------------------------------------------- #
    md(r"""
## {n} · Fine-Tuning — trainiere *deine eigene* Stimme

Few-Shot ist sofort, aber an die Referenz gebunden. Für die **letzten Prozente Ähnlichkeit**
trainierst du das Modell auf einem Datensatz deiner Stimme. Das ist ein **GPU-Batch-Job**
(läuft nicht live hier), aber die komplette Pipeline ist echter Code — so gehst du vor:

1. **Aufnehmen.** Lies das vollständige Lese-Skript ein:
   [`data/recording_script.md`](../data/recording_script.md) — **814 Sätze, ~44 min, 3 Sessions**
   (Harvard- + expressive + konversationelle + technische Sätze). Pro Satz ein Clip, ruhiger Raum,
   konstanter Mikro-Abstand, **24 kHz mono, SNR ≥ 30 dB**. Beginne jede Session mit dem Kalibriersatz.
2. **Aufbereiten** — `voxclone-prep` (entrauschen → VAD-schneiden → 24 kHz → Whisper-Transkript →
   validieren). Ergebnis: ein sauberes Manifest + Train/Held-out-Split + dein Referenzclip.
3. **Trainieren** — je Engine in ihrer Conda-Env: `voxclone-train` (XTTS-v2: 15 Epochen, lr 5e-6;
   F5-TTS: 100 Epochen, lr 1e-5). Schritt-für-Schritt im
   [`docs/RUNBOOK_finetune.md`](../docs/RUNBOOK_finetune.md).
4. **Bewerten** — `voxclone-eval` scort dein fine-getuntes Modell auf den Held-out-Sätzen mit
   genau der Formel aus Abschnitt 3 (Ähnlichkeit · UTMOS · WER), inkl. der ehrlichen
   real↔real-Obergrenze aus Abschnitt 4.

> Der entscheidende Befund aus Abschnitt 3 gilt auch für dich: **die richtige Engine zu wählen
> bringt mehr als Fine-Tuning der falschen** — F5 zero-shot schlägt XTTS voll fine-getunt. Fang
> also mit Few-Shot (oben) an und tune erst, wenn du die letzten Prozente brauchst.
""".format(n=7 if is_prof else 6))

    # ----------------------------------------------------------------------- #
    # Grenzen                                                                  #
    # ----------------------------------------------------------------------- #
    md(r"""
## {n} · Grenzen & ehrliche Einordnung

Die Ergebnisse sind **richtungsweisend, nicht endgültig** — bewusst transparent:

- **N = 6, keine Konfidenzintervalle.** Score-Abstände unter ~0,02 sind Gleichstände
  (F5 fine-getunt vs zero-shot: +0,0053 — ein Gleichstand; *nicht* behaupten, Fine-Tuning helfe F5).
- **Obergrenze ist ein optimistisches Same-Session-Paar** (zwei Hälften einer Aufnahme), also lies
  „90 % der Obergrenze" als *relative* Position zwischen Engines, nicht als „90 % nicht
  unterscheidbar von einem Menschen".
- **Same-Session-Bewertung** — Held-out-Clips stammen aus derselben Aufnahmesession wie das Training;
  gemessen wird In-Session-Treue, nicht Generalisierung auf neuen Tag / Mikro / Raum.
- **UTMOS ist gesättigt** — alle sechs Klone liegen über dem echten UTMOS (3,295); die 0,50-
  gewichtete Natürlichkeit trennt kaum. „Poliert", nicht „menschlicher als der Mensch".
- **Kein menschlicher Hörtest** — das Urteil ruht auf automatischen Proxys (ECAPA + UTMOS + WhisperX).

**Was jede Einschränkung übersteht:** F5 führt in beiden Modi, und **F5 zero-shot schlägt XTTS
fine-getunt** mit großem Abstand — die richtige Engine zu wählen zählt mehr als das Fine-Tuning der
schwächeren. Voller Bericht: [`docs/finetune-vs-zeroshot-report.md`](../docs/finetune-vs-zeroshot-report.md).
""".format(n=8 if is_prof else 7))

    # ----------------------------------------------------------------------- #
    # Troubleshooting + Quellen                                               #
    # ----------------------------------------------------------------------- #
    md(r"""
## {n} · Troubleshooting & Quellen

- **Erster Clip langsam (~60–90 s).** Das Modell lädt auf die GPU; danach ist dieselbe Stimme schnell.
- **„Falscher Kernel"?** Kernel-Menü → **`VoxClone (.venv)`**, dann Run All.
- **Eine Stimme hängt?** „Restart voice", dann erneut Generate.
- Erzeugte Clips liegen unter `notebooks/clone_outputs/`.
- **Setup auf einer frischen Maschine:** [`docs/SETUP.md`](../docs/SETUP.md).

**Quellen (echter Code & Daten, kein Paraphrasieren):**
`experiments/ab_eval_finetune_results.json` · `voxclone.eval.picker.combined_score` ·
`voxclone.eval.similarity` · `voxclone.clone.studio` · `voxclone.prep` · `voxclone.train` ·
Lese-Skript [`data/recording_script.md`](../data/recording_script.md) ·
Runbook [`docs/RUNBOOK_finetune.md`](../docs/RUNBOOK_finetune.md) ·
Bericht [`docs/finetune-vs-zeroshot-report.md`](../docs/finetune-vs-zeroshot-report.md).
""".format(n=9 if is_prof else 8))

    return cells


# Public build is the module-level default (committed to the repo + used by tests).
cells = _build_cells("public")


def build_notebook(audience: str = "public") -> nbf.NotebookNode:
    return new_notebook(cells=_build_cells(audience), metadata={
        "kernelspec": {"name": "voxclone", "display_name": "VoxClone (.venv)", "language": "python"},
        "language_info": {"name": "python"},
    })


def write_notebook(path: str | None = None, audience: str = "public") -> Path:
    out = Path(path) if path else Path(__file__).resolve().parent / "voice_cloning_study.ipynb"
    nbf.write(build_notebook(audience), str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audience", choices=AUDIENCES, default="public",
                    help="public = repo (no author audio/weights); professor = ZIP (keeps hear-my-voice).")
    ap.add_argument("--out", default=None, help="output .ipynb path (default: notebooks/voice_cloning_study.ipynb)")
    args = ap.parse_args()
    out = write_notebook(args.out, audience=args.audience)
    print(f"wrote {out}  (audience={args.audience}, {len(_build_cells(args.audience))} cells)")


if __name__ == "__main__":
    main()
