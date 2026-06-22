import shutil
import subprocess
from pathlib import Path

from voxclone.prep.manifest import ClipRecord, read_manifest
from voxclone.train.base import TrainResult
from voxclone.common.logging import get_logger

log = get_logger("train.f5")

_HEADER = "audio_file|text"

# Where finetune mode expects the pretrained vocab, relative to <data_root>. f5_tts'
# prepare_csv_wavs (finetune mode) ASSERTS this exists and copies it into the prepared
# dataset dir. It is NOT auto-downloaded by the trainer.
_PRETRAIN_VOCAB_REL = Path("Emilia_ZH_EN_pinyin") / "vocab.txt"
_PRETRAIN_VOCAB_SRC = "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt"


def manifest_to_f5(records: list[ClipRecord], csv_path) -> Path:
    """F5-TTS metadata CSV for prepare_csv_wavs.py: UTF-8, header EXACTLY ``audio_file|text``,
    col1 = ABSOLUTE wav path, col2 = verbatim transcript. f5 raises on non-absolute paths;
    we reject early. Empty-text rows are skipped.
    (prepare_csv_wavs.py does NOT resample — wavs must already be 24 kHz mono.)"""
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [_HEADER]
    for r in records:
        text = r.text.strip()
        if not text:
            continue
        if not Path(r.audio_path).is_absolute():
            raise ValueError(f"f5 requires absolute wav paths, got: {r.audio_path}")
        lines.append(f"{r.audio_path}|{text}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


class F5Trainer:
    """F5-TTS fine-tune adapter (TrainAdapter), GPU-free-testable via an injectable ``runner``.

    f5_tts (1.1.x) has NO output/save-dir flag; the prepared-dataset dir and the checkpoint
    dir BOTH derive solely from ``dataset_name``, anchored at roots that live next to the
    installed ``f5_tts`` package (``files("f5_tts")/../../{data,ckpts}``). Two GPU steps run
    inside the ``f5tts`` conda env through ``runner``:

      (A) ``python -m f5_tts.train.datasets.prepare_csv_wavs <metadata.csv>
          <data_root>/{name}_pinyin`` — the FIRST positional MUST be the metadata.csv FILE
          (is_csv_wavs_format checks is_file() + suffix=='.csv'); finetune is the DEFAULT mode
          (the ONLY flag is --pretrain, there is NO --finetune flag), and that default mode
          requires & copies the pretrained Emilia_ZH_EN_pinyin vocab.txt while writing the
          Arrow dataset to ``<data_root>/{name}_pinyin``.
      (B) ``accelerate launch -m f5_tts.train.finetune_cli --exp_name F5TTS_v1_Base
          --dataset_name {name} --finetune --tokenizer pinyin ...`` — the BARE ``dataset_name``
          is passed; ``load_dataset`` re-adds the ``_pinyin`` suffix. The trainer writes
          ``<ckpts_root>/{name}/model_last.pt``.

    After (B) the real ``model_last.pt`` and the prepared ``vocab.txt`` are copied into
    ``out_dir`` so the registry/serve path (which feeds ``out_dir`` to ``F5TTS(ckpt_file=...,
    vocab_file=...)``) finds real FILES. ``--tokenizer pinyin`` is kept for English (it matches
    the pretrained embedding). There is intentionally NO --output_dir/--save_dir/--ref_audio.

    ``data_root``/``ckpts_root`` default to the f5_tts-relative roots; when None they are
    resolved at ``train()`` time via a ``conda run`` into the f5tts env (the trainer's own
    .venv cannot import f5_tts). Inject explicit tmp roots in tests to stay subprocess-free.
    """

    def __init__(self, conda_env: str = "f5tts", dataset_name: str = "danil",
                 data_root: str | None = None, ckpts_root: str | None = None,
                 runner=subprocess.run):
        self.conda_env = conda_env
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.ckpts_root = ckpts_root
        self.runner = runner

    def _resolve_roots(self) -> tuple[Path, Path]:
        """Return (data_root, ckpts_root) as Paths, resolving the f5_tts-relative defaults
        through the conda env when they were not injected."""
        if self.data_root is not None and self.ckpts_root is not None:
            return Path(self.data_root), Path(self.ckpts_root)
        base = self._resolve_f5_base()  # pragma: no cover
        data_root = Path(self.data_root) if self.data_root is not None else base / "data"  # pragma: no cover
        ckpts_root = Path(self.ckpts_root) if self.ckpts_root is not None else base / "ckpts"  # pragma: no cover
        return data_root, ckpts_root  # pragma: no cover

    def _resolve_f5_base(self) -> Path:  # pragma: no cover
        # f5_tts is not importable from the trainer's .venv; ask the f5tts env for the root
        # that contains its data/ and ckpts/ dirs (files("f5_tts")/../..).
        cmd = ["conda", "run", "-n", self.conda_env, "python", "-c",
               "from importlib.resources import files; "
               "print(files('f5_tts').joinpath('../..'))"]
        res = self.runner(cmd, capture_output=True, text=True)
        if getattr(res, "returncode", 0) != 0:
            raise RuntimeError(f"f5 root resolution failed: {cmd}")
        return Path(res.stdout.strip())

    def train(self, manifest_path: str, out_dir: str, config: dict) -> TrainResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        data_root, ckpts_root = self._resolve_roots()

        # Vocab prereq: finetune mode requires the pretrained Emilia_ZH_EN_pinyin vocab.txt.
        pretrain_vocab = data_root / _PRETRAIN_VOCAB_REL
        if not pretrain_vocab.exists():
            raise FileNotFoundError(
                f"F5 finetune mode requires the pretrained vocab at {pretrain_vocab}. "
                f"Fetch it manually from {_PRETRAIN_VOCAB_SRC} (the trainer does not "
                f"auto-download it).")

        records = read_manifest(manifest_path)
        # metadata.csv goes into a staging input dir that prepare_csv_wavs reads from.
        staging = out / "input"
        csv_path = manifest_to_f5(records, staging / "metadata.csv")

        # f5 LOADS the dataset from <data_root>/{name}_pinyin (load_dataset adds _pinyin),
        # so prepare must WRITE the Arrow dataset there.
        prepared_dir = data_root / f"{self.dataset_name}_pinyin"

        # (A) prepare Arrow dataset. FIRST positional is the metadata.csv FILE (NOT the
        # staging dir): is_csv_wavs_format requires Path(p).is_file() and suffix=='.csv'.
        # Finetune is the DEFAULT (which copies the pretrained vocab.txt); the ONLY mode flag
        # is --pretrain, and there is NO --finetune flag (passing it is an argparse error).
        prepare = ["conda", "run", "-n", self.conda_env,
                   "python", "-m", "f5_tts.train.datasets.prepare_csv_wavs",
                   str(csv_path), str(prepared_dir)]
        self._run(prepare, "prepare_csv_wavs")

        # (B) accelerate finetune. --dataset_name is the BARE name (finetune_cli re-adds
        # _pinyin via load_dataset). Only real finetune_cli flags are used; the checkpoint
        # dir derives from --dataset_name (NO save/output/ref flag exists).
        epochs = int(config.get("epochs", 60))
        finetune = ["conda", "run", "-n", self.conda_env,
                    "accelerate", "launch",
                    "-m", "f5_tts.train.finetune_cli",
                    "--exp_name", "F5TTS_v1_Base",
                    "--dataset_name", self.dataset_name,
                    "--finetune", "--tokenizer", "pinyin",
                    "--learning_rate", str(config.get("learning_rate", 1e-5)),
                    "--batch_size_per_gpu", str(config.get("batch_size_per_gpu", 3200)),
                    "--batch_size_type", "frame",
                    "--epochs", str(epochs),
                    "--grad_accumulation_steps", str(config.get("grad_accumulation_steps", 1)),
                    # f5's default num_warmup_updates (20000) is tuned for the full-Emilia
                    # finetune; on a small single-speaker set that warmup never completes, so
                    # the LR stays ~0 and nothing is learned. Default small here + configurable.
                    "--num_warmup_updates", str(config.get("num_warmup_updates", 300)),
                    "--save_per_updates", str(config.get("save_per_updates", 50000)),
                    "--last_per_updates", str(config.get("last_per_updates", 1000))]
        self._run(finetune, "finetune_cli")

        # The real outputs live under the f5-relative roots; copy the ckpt FILE + prepared
        # vocab into out_dir so the registry/serve path (which uses out_dir) finds real files.
        ckpt_src = ckpts_root / self.dataset_name / "model_last.pt"
        if not ckpt_src.exists():
            raise FileNotFoundError(
                f"F5 finetune produced no checkpoint at {ckpt_src} — expected the trainer "
                f"to write model_last.pt under <ckpts_root>/{self.dataset_name}/.")
        shutil.copy2(ckpt_src, out / "model_last.pt")

        vocab_src = prepared_dir / "vocab.txt"
        if not vocab_src.exists():
            raise FileNotFoundError(
                f"F5 prepare produced no vocab at {vocab_src} — expected prepare_csv_wavs "
                f"--finetune to copy the pretrained vocab.txt into the prepared dataset dir.")
        shutil.copy2(vocab_src, out / "vocab.txt")

        return TrainResult(checkpoint_dir=str(out), steps=epochs)

    def _run(self, cmd, label):
        log.info("running %s: %s", label, " ".join(cmd))
        res = self.runner(cmd)
        if getattr(res, "returncode", 0) != 0:
            raise RuntimeError(f"f5 {label} failed: {cmd}")
