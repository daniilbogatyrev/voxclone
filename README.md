# voxclone

**Eine Stimme klonen — und *messen*, ob es funktioniert.**

Persönliches, **nicht-kommerzielles** Projekt: Es klont **eine einwilligende Stimme**
(meine, mit Einwilligung für dieses Projekt aufgenommen) und vergleicht vier quelloffene
TTS-Engines auf derselben Stimme — objektiv gescort, nicht nach Gefühl. Zugleich ist es eine
**Anleitung, mit der du deine eigene Stimme klonst** (Few-Shot und Fine-Tuning).

## Was es ist

Eine saubere `voxclone`-Bibliothek mit drei Oberflächen:

- einer **CLI** für die Daten- und Trainings-Pipeline,
- einer **FastAPI- + Vanilla-JS-Web-App** zum interaktiven Klonen,
- **dem einen Notebook** [`notebooks/voice_cloning_study.ipynb`](notebooks/voice_cloning_study.ipynb),
  das dich durch alles führt: Pipeline, Ergebnisse, die „≈ 90 %"-Zahl und das Klonen *deiner
  eigenen* Stimme.

Darunter liegt ein forschungsnaher Mehr-Engine-Vergleich (XTTS-v2, F5-TTS, Chatterbox,
GPT-SoVITS), *zero-shot* und *fine-getunt*.

## Ergebnis in einem Satz

**F5-TTS gewinnt** und bildet die Stimme zu **≈ 90 %** der ehrlichen real↔real-Obergrenze ab.
Fine-Tuning hilft **XTTS** stark, **F5** ist schon gesättigt — und **F5 ohne Training schlägt
XTTS voll fine-getunt**. Die Lehre: *die richtige Engine zu wählen zählt mehr als das
Fine-Tuning der falschen.* Voller Bericht:
[`docs/finetune-vs-zeroshot-report.md`](docs/finetune-vs-zeroshot-report.md).

## Was bewusst **nicht** im Repo ist

Aus Datenschutz- und Einwilligungsgründen enthält dieses öffentliche Repo **keine Stimm-Audios
und keine fine-getunten Gewichte** von mir. Die gemessenen Ergebnisse (Tabellen / Diagramme)
sind enthalten, das Audio nicht. Das Notebook lässt dich stattdessen **deine eigene Stimme**
klonen. Die Modelle (XTTS-Basis, F5-Deutsch usw.) lädst du frei von Hugging Face — siehe
[`docs/SETUP.md`](docs/SETUP.md).

## Schnellstart

Frischer Clone → Projekt-venv:

```bash
git clone https://github.com/<user>/voxclone.git && cd voxclone
uv sync --extra dev
.venv/bin/python -m ipykernel install --user --name voxclone --display-name "VoxClone (.venv)"
```

Das ist nur die Projekt-venv. Um das Notebook wirklich laufen zu lassen, brauchst du die
per-Engine-Conda-Envs, die Modelle und einen Referenzclip — eine frische Kopie hat davon
**nichts** (alles gitignored). Die vollständige, kopierbare Anleitung für eine frische Maschine
ist **[`docs/SETUP.md`](docs/SETUP.md)**.

**Hardware:** Die Generierung braucht eine **NVIDIA-CUDA-GPU** (Stack auf torch 2.11.0+cu128,
Blackwell/sm_120 gepinnt). Ohne GPU laufen nur die GPU-freien Analyse-Teile.

## Das eine Notebook

[`notebooks/voice_cloning_study.ipynb`](notebooks/voice_cloning_study.ipynb) öffnen (VS Code,
Kernel **`VoxClone (.venv)`**, cwd = Repo-Wurzel) und **Run All**. Es führt dich durch:

1. **Den Trainingsprozess** — von der Aufnahme zum Klon (die Pipeline).
2. **Die Ergebnisse** — die Rangliste, jede Zahl aus echtem Code neu hergeleitet, plus Diagramme.
3. **Die Schlüsselzahl** — wie „≈ 90 %" berechnet wird (ECAPA-Ähnlichkeit / Obergrenze).
4. **Few-Shot** — *deine* Stimme aus einem kurzen Clip klonen (kein Training).
5. **Fine-Tuning** — *deine* Stimme end-to-end trainieren (Aufnahme → prep → train → eval).

Die Analyse-Abschnitte laufen **GPU-frei**; die Audio-Abschnitte brauchen das volle Setup und
degradieren sonst freundlich. Du brauchst nie ein Terminal — das Notebook startet die
Engine-Server selbst.

## Lese-Skript & Few-Shot-Satz

Für eine zero-shot-Referenz liest du diesen **Few-Shot-Satz** (Kalibriersatz) ein, ~6–12 s,
mono WAV:

> *„This is my natural speaking voice, calm, clear, and steady, as I read these few lines aloud today."*

Das vollständige Lese-Skript fürs Fine-Tuning (**814 Sätze, ~44 min, 3 Sessions**) liegt in
[`data/recording_script.md`](data/recording_script.md).

## Pipeline (CLI)

```
capture → prep → split → train → eval
```

CLI-Einstiegspunkte (in `pyproject.toml`, Code in `src/scripts/`):
`voxclone-{prep,train,eval,serve,capture,verify,import}`. Beispiel — Referenzaudio aufbereiten:

```bash
uv run voxclone-prep --raw data/raw --out data/processed --config configs/default.yaml
```

## Engines & Sprachen

XTTS-v2, F5-TTS, Chatterbox, GPT-SoVITS — jede in **eigener Conda-Env**
(`xtts`/`f5tts`/`chatterbox`/`gptsovits`), getrennt von der `.venv`; das Notebook spricht per
HTTP mit ihnen. **Deutsch** (cross-lingual, englische Referenz → deutsche Ausgabe) geht mit
XTTS (nativ), F5 (deutscher Checkpoint) und Chatterbox (multilingual); **GPT-SoVITS kann kein
Deutsch** und wird dafür übersprungen.

## Tests

```bash
uv run --extra dev pytest -q
```

344 GPU-freie Tests (schwere Modell-Libs liegen hinter Lazy Imports / injizierbaren Nähten).

## Dokumentation

- [`docs/SETUP.md`](docs/SETUP.md) — frische Maschine → Notebook läuft (hier starten).
- [`docs/HANDOFF.md`](docs/HANDOFF.md) — Projektstand + per-Engine-Installations-Runbooks.
- [`docs/RUNBOOK_finetune.md`](docs/RUNBOOK_finetune.md) — die Fine-Tunes erzeugen.
- [`docs/finetune-vs-zeroshot-report.md`](docs/finetune-vs-zeroshot-report.md) — Ergebnisse/Vergleich.

## Ethik / Einwilligung

Jede Stimme in diesem Projekt ist die des Eigentümers, mit ausdrücklicher Einwilligung
aufgenommen. **Klone nur Stimmen, für die du die Erlaubnis der sprechenden Person hast.**
Nutze dieses Repo nicht für etwas außerhalb dieses persönlichen, nicht-kommerziellen Projekts.
