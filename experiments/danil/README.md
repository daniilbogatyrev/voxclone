# Danil — voice clone

Per-person workspace for cloning Danil's English voice.

## Layout
- `reference/` — **upload your recorded reference clips here.** Clean WAVs of Danil
  speaking (record locally, then drop the files in this folder). These are the
  speaker source the model conditions on.
- `experiment/` — generated clone outputs land here (`NNN_<model>_<mode>.wav`,
  normalized to −16 LUFS, ready to play).

`*.wav` files are gitignored (binaries, local-only). The canonical run record stays
in `experiments/log.jsonl` (one JSON line per run), tagged `"speaker":"danil"`.

## Recording the reference (tips)
- Quiet room, consistent distance from the mic, natural calm pace.
- 10–30 s per clip is plenty for zero-shot; mono WAV preferred.
- Note the exact transcript of each reference clip — GPT-SoVITS v2Pro **requires**
  a non-empty `prompt_text` equal to the words actually spoken in the reference.

## Next
Once a reference clip is in `reference/`, tell me its filename + transcript and I'll
run a zero-shot clone (v2Pro, `cut0`) into `experiment/` and log it.
