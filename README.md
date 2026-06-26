# voxclone

**Eine Stimme klonen — und *messen*, ob es funktioniert.**

Persönliches, **nicht-kommerzielles** Projekt: Es klont **eine einwilligende Stimme**
(meine, mit Einwilligung für dieses Projekt aufgenommen) und vergleicht vier quelloffene
TTS-Engines auf derselben Stimme — objektiv gescort. Zugleich ist es eine
**Anleitung, mit der du deine eigene Stimme klonst** (Few-Shot und Fine-Tuning).

## Was es ist

Eine saubere `voxclone`-Bibliothek mit drei Oberflächen:

- einer **CLI** für die Daten- und Trainings-Pipeline,
- einer **FastAPI- + Vanilla-JS-Web-App** zum interaktiven Klonen,
- **dem einen Notebook** [`notebooks/voice_cloning_study.ipynb`](notebooks/voice_cloning_study.ipynb),
  das dich durch alles führt: Pipeline, Ergebnisse, die Ähnlichkeits­messung und das Klonen *deiner
  eigenen* Stimme.

Darunter liegt ein forschungsnaher Mehr-Engine-Vergleich (XTTS-v2, F5-TTS, Chatterbox,
GPT-SoVITS), *zero-shot* und *fine-getunt*.

## Ergebnis in einem Satz

**F5-TTS gewinnt** den Engine-Vergleich — es klingt der echten Stimme am ähnlichsten (gemessen
als Sprecher-Ähnlichkeit relativ zu einer ehrlichen real↔real-Obergrenze). Fine-Tuning hilft
**XTTS** stark, **F5** ist schon nahe an seiner Grenze — und **F5 ohne Training schlägt XTTS
voll fine-getunt**. Die Lehre: *die richtige Engine zu wählen zählt mehr als das Fine-Tuning der
falschen.*

Die konkreten Messwerte hier stammen von **meiner** Stimme — wie nah es bei **deiner** Stimme
kommt, hängt von der Aufnahme ab und **misst du selbst**: dieselbe Bewertungs-Pipeline berechnet
die Scores für jede Stimme, die du klonst. Voller Bericht:
[`docs/finetune-vs-zeroshot-report.md`](docs/finetune-vs-zeroshot-report.md).

## Status & Ausblick

Dies ist ein **laufendes Studienprojekt** — der aktuelle Stand (Engine-Vergleich, Few-Shot-
und Fine-Tuning-Klon, Bewertung) ist abgeschlossen und reproduzierbar, aber das Projekt wächst
weiter. Mögliche nächste Schritte, evtl. in einem weiteren Semester:

- **Lip-Syncing** — die geklonte Stimme mit lippensynchronem Video koppeln (Audio → Mundbewegung).
- **Weitere Sprachen** — über Englisch/Deutsch hinaus (z. B. Französisch, Spanisch), inkl.
  cross-lingualer Bewertung.
- **Größerer, mehrtägiger Hörtest** und ein menschliches Urteil zusätzlich zu den automatischen Metriken.

Das sind **Ideen, keine Zusagen**. Die Architektur (eine `voxclone`-
Bibliothek, per-Engine-Adapter, ein gemeinsames Bewertungs-Frontend) ist bewusst so gebaut, dass
sich solche Erweiterungen anbauen lassen.

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
Blackwell/sm_120 gepinnt).
## Das eine Notebook

[`notebooks/voice_cloning_study.ipynb`](notebooks/voice_cloning_study.ipynb) öffnen (VS Code,
Kernel **`VoxClone (.venv)`**, cwd = Repo-Wurzel) und **Run All**. Es führt dich durch:

1. **Den Trainingsprozess** — von der Aufnahme zum Klon (die Pipeline).
2. **Die Ergebnisse** — die Engine-Rangliste mit Diagrammen, nachvollziehbar aus den Messwerten hergeleitet.
3. **Die Ähnlichkeits-Kennzahl** — wie gemessen wird, wie nah die geklonte Stimme der echten kommt (ECAPA-Ähnlichkeit / Obergrenze).
4. **Few-Shot** — *deine* Stimme aus einem kurzen Clip klonen (kein Training).
5. **Fine-Tuning** — *deine* Stimme end-to-end trainieren (Aufnahme → prep → train → eval).

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
