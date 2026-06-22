# Experiments

Voice-clone generation runs. Each run = one JSON line in `log.jsonl` plus its audio files here.

## Layout
- `NNN_<model>_<mode>.wav` — the generated clip (normalized to −16 LUFS, ready to play).
- `NNN_reference.wav` — the reference clip used for that run.
- `log.jsonl` — canonical record, one JSON object per run (model, reference, prompt_text, target text, params, output, ASR check, notes). **Git-tracked.** The `*.wav` files are gitignored (binaries; local-only).

## Play a clip
```bash
pw-play experiments/001_v2pro_zeroshot.wav
# or
ffplay -autoexit -nodisp experiments/001_v2pro_zeroshot.wav
```

## Runs so far
- **001** — v2Pro zero-shot: _"This is a cloned version of my voice, created by a computer."_ → `001_v2pro_zeroshot.wav`. First working clone (v2 base collapsed to silence on split fragments; v2Pro + no-split `cut0` fixed it).
