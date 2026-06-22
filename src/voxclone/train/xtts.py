import random
import subprocess
from pathlib import Path

from voxclone.prep.manifest import ClipRecord, read_manifest
from voxclone.train.base import TrainResult
from voxclone.common.logging import get_logger

log = get_logger("train.xtts")

# XTTS loader silently DROPS (does not truncate) clips longer than max_wav_length
# samples (255995 ~= 11.6 s @ 22.05 kHz) and texts longer than max_text_length chars.
SAMPLE_RATE = 22050
MAX_WAV_LENGTH = 255995          # samples
MAX_TEXT_LENGTH = 200            # chars
_HEADER = "audio_file|text|speaker_name"


def manifest_to_xtts(records: list[ClipRecord], out_dir, speaker: str = "danil",
                     eval_fraction: float = 0.15, seed: int = 0,
                     max_wav_length: int = MAX_WAV_LENGTH,
                     max_text_length: int = MAX_TEXT_LENGTH) -> tuple[Path, Path]:
    """Coqui-format metadata for the GPTTrainer recipe (NOT ljspeech).

    Writes metadata_train.csv + metadata_eval.csv under ``out_dir``, each with a header
    row ``audio_file|text|speaker_name`` then pipe-delimited rows where col1 = ABSOLUTE
    ClipRecord.audio_path (so the recipe runs with root_path=""), col2 = text, col3 =
    speaker. Filters empty text, clips longer than ``max_wav_length`` samples (XTTS
    silently drops, not truncates) and texts longer than ``max_text_length`` chars.
    The eval split (``metadata_eval.csv``) is XTTS's in-training LOSS split, carved
    deterministically by ``seed`` — it is DISTINCT from the project held-out set.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    max_wav_seconds = max_wav_length / SAMPLE_RATE
    rows = []
    for r in records:
        text = r.text.strip()
        if not text or r.duration > max_wav_seconds or len(text) > max_text_length:
            continue
        rows.append(f"{r.audio_path}|{text}|{speaker}")
    random.Random(seed).shuffle(rows)
    n_eval = int(round(len(rows) * eval_fraction))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    train_csv, eval_csv = out / "metadata_train.csv", out / "metadata_eval.csv"
    train_csv.write_text(_HEADER + "\n" + "\n".join(train_rows) + "\n", encoding="utf-8")
    eval_csv.write_text(_HEADER + "\n" + "\n".join(eval_rows) + "\n", encoding="utf-8")
    return train_csv, eval_csv


class XTTSTrainer:
    """Fine-tune XTTS-v2 via the coqui GPTTrainer recipe.

    Writes the two coqui CSVs via ``manifest_to_xtts`` then shells out (through the
    injectable ``runner``) to the thin ``voxclone.train.xtts_recipe`` driver INSIDE the
    ``xtts`` conda env. Honors small-dataset overrides from ``config`` rather than the
    recipe's big-corpus defaults (whose grad_accum=84 is wrong for this set): epochs 6-15
    (default 10), batch_size 3-6 (default 4), grad_accum 1-4 (default 2), lr 5e-6,
    mixed_precision True, max_wav_length 255995, max_text_length 200.
    """

    def __init__(self, xtts_root: str, conda_env: str = "xtts",
                 speaker: str = "danil", runner=subprocess.run):
        self.xtts_root = xtts_root
        self.conda_env = conda_env
        self.speaker = speaker
        self.runner = runner

    def train(self, manifest_path: str, out_dir: str, config: dict) -> TrainResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        records = read_manifest(manifest_path)
        train_csv, eval_csv = manifest_to_xtts(records, out, speaker=self.speaker)
        epochs = int(config.get("epochs", 10))
        # Invoke the recipe by FILE PATH, not `-m voxclone.train.xtts_recipe`: the xtts
        # conda env doesn't have voxclone installed, and importing the voxclone.train
        # package would pull in pydantic (also absent there). xtts_recipe.py is self-
        # contained (no voxclone imports), so running it by path needs neither.
        recipe = str(Path(__file__).with_name("xtts_recipe.py"))
        cmd = [
            "conda", "run", "-n", self.conda_env, "python",
            recipe,
            "--model_dir", self.xtts_root,
            "--train_csv", str(train_csv), "--eval_csv", str(eval_csv),
            "--out", str(out),
            "--epochs", str(epochs),
            "--batch_size", str(config.get("batch_size", 4)),
            "--grad_accum", str(config.get("grad_accum", 2)),
            "--lr", str(config.get("lr", 5e-6)),
            "--mixed_precision", str(config.get("mixed_precision", True)),
            "--max_wav_length", str(config.get("max_wav_length", MAX_WAV_LENGTH)),
            "--max_text_length", str(config.get("max_text_length", MAX_TEXT_LENGTH)),
        ]
        log.info("running: %s", " ".join(cmd))
        res = self.runner(cmd)
        if getattr(res, "returncode", 0) != 0:
            raise RuntimeError(f"xtts training failed: {cmd}")
        return TrainResult(checkpoint_dir=str(out), steps=epochs)
