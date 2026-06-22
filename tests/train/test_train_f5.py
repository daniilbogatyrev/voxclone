from pathlib import Path

import pytest
from voxclone.prep.manifest import ClipRecord, write_manifest
from voxclone.train.f5 import manifest_to_f5, F5Trainer
from voxclone.train.base import TrainResult


def _rec(path, text):
    return ClipRecord(audio_path=path, text=text, duration=2.0,
                      transcript_confidence=0.9, clipped_fraction=0.0, category="harvard")


# ---- manifest_to_f5 (pipe CSV audio_file|text) ----

def test_emits_required_header_abs_paths_pipe_delimited(tmp_path):
    csv_path = manifest_to_f5([_rec("/abs/a.wav", "hello there"),
                               _rec("/abs/b.wav", "good morning")], tmp_path / "metadata.csv")
    lines = (tmp_path / "metadata.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "audio_file|text"          # EXACT header f5 requires
    assert lines[1] == "/abs/a.wav|hello there"
    assert lines[2] == "/abs/b.wav|good morning"
    assert str(csv_path).endswith("metadata.csv")


def test_skips_empty_text(tmp_path):
    manifest_to_f5([_rec("/abs/a.wav", "hi"), _rec("/abs/b.wav", "  ")], tmp_path / "m.csv")
    body = [ln for ln in (tmp_path / "m.csv").read_text().splitlines()[1:] if ln]
    assert len(body) == 1


def test_rejects_non_absolute_paths(tmp_path):
    with pytest.raises(ValueError):
        manifest_to_f5([_rec("relative/a.wav", "hi")], tmp_path / "m.csv")


# ---- F5Trainer (prepare_csv_wavs + accelerate finetune_cli) ----

def _write_manifest(p):
    write_manifest(p, [
        ClipRecord(audio_path="/abs/a.wav", text="hello", duration=2.0,
                   transcript_confidence=0.9, clipped_fraction=0.0, category="harvard"),
    ])


def _make_roots(tmp_path):
    """Create tmp data_root + ckpts_root and a pretrained-vocab prereq under data_root."""
    data_root = tmp_path / "data"
    ckpts_root = tmp_path / "ckpts"
    data_root.mkdir(parents=True, exist_ok=True)
    ckpts_root.mkdir(parents=True, exist_ok=True)
    # pretrained vocab prerequisite that finetune mode requires
    pretrain_dir = data_root / "Emilia_ZH_EN_pinyin"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    (pretrain_dir / "vocab.txt").write_text("a\nb\n", encoding="utf-8")
    return data_root, ckpts_root


def _fake_runner(calls, data_root, ckpts_root, dataset_name="danil"):
    """Records commands, simulates success, and materialises the real artifacts the
    f5 pipeline would have produced: the prepared dataset's vocab.txt and the
    trainer's model_last.pt checkpoint, both under the injected tmp roots."""
    def run(cmd, **kw):
        calls.append(cmd)
        flat = " ".join(cmd)
        if "prepare_csv_wavs" in flat:
            # mirror prepare_csv_wavs' contract: ensure the metadata.csv FILE the trainer
            # passes as the first positional exists, then write the prepared dataset out dir.
            mod_idx = cmd.index("f5_tts.train.datasets.prepare_csv_wavs")
            csv_arg = Path(cmd[mod_idx + 1])
            csv_arg.parent.mkdir(parents=True, exist_ok=True)
            if not csv_arg.exists():
                csv_arg.write_text("audio_file|text\n", encoding="utf-8")
            prepared = Path(data_root) / f"{dataset_name}_pinyin"
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "vocab.txt").write_text("a\nb\n", encoding="utf-8")
        elif "finetune_cli" in flat:
            ckpt_dir = Path(ckpts_root) / dataset_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "model_last.pt").write_bytes(b"FAKE_CKPT")

        class R:
            returncode = 0
        return R()
    return run


def test_trainer_runs_prepare_then_finetune(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    data_root, ckpts_root = _make_roots(tmp_path)
    calls = []
    trainer = F5Trainer(conda_env="f5tts", dataset_name="danil",
                        data_root=str(data_root), ckpts_root=str(ckpts_root),
                        runner=_fake_runner(calls, data_root, ckpts_root))
    result = trainer.train(str(manifest), str(tmp_path / "out"),
                           {"epochs": 60, "learning_rate": 1e-5})
    assert isinstance(result, TrainResult)
    assert len(calls) == 2                                    # (A) prepare, then (B) finetune
    prep, finetune = " ".join(calls[0]), " ".join(calls[1])

    # (A) prepare_csv_wavs: finetune is the DEFAULT (no --finetune flag exists), the FIRST
    # positional MUST be the metadata.csv FILE, prepared OUT dir is <data_root>/{name}_pinyin
    assert "prepare_csv_wavs" in prep
    assert "--finetune" not in prep                           # NO --finetune flag in argparse (finetune is default)
    assert "--pretrain" not in prep                           # finetune mode copies pretrained vocab
    # first positional after the module name must be a .csv FILE that exists on disk
    mod_idx = calls[0].index("f5_tts.train.datasets.prepare_csv_wavs")
    csv_arg = calls[0][mod_idx + 1]
    assert csv_arg.endswith(".csv")                           # a FILE, not a staging dir
    assert Path(csv_arg).is_file()                            # the metadata.csv actually exists
    out_arg = calls[0][mod_idx + 2]                           # second positional = prepared OUT dir
    assert out_arg == str(Path(data_root) / "danil_pinyin")   # BARE name + _pinyin suffix here

    # (B) finetune_cli: BARE --dataset_name (load_dataset adds _pinyin), real flags only
    assert "finetune_cli" in finetune
    assert "--finetune" in finetune and "--tokenizer" in finetune and "pinyin" in calls[1]
    assert "--dataset_name" in finetune and "danil" in calls[1]
    assert "danil_pinyin" not in calls[1]                     # the _pinyin suffix is NOT passed to finetune_cli
    assert "--exp_name" in finetune and "F5TTS_v1_Base" in calls[1]
    assert "--learning_rate" in finetune
    assert "1e-05" in calls[1] or "1e-5" in calls[1]          # learning_rate 1e-5
    assert "--batch_size_per_gpu" in finetune and "3200" in calls[1]
    assert "--batch_size_type" in finetune and "frame" in calls[1]
    assert "--grad_accumulation_steps" in finetune
    assert "--epochs" in finetune and "60" in calls[1]
    # small warmup so the LR actually ramps on a tiny single-speaker set (f5's 20000
    # default never completes -> nothing learned); + periodic last-checkpoint saves.
    assert "--num_warmup_updates" in finetune and "300" in calls[1]   # default, not 20000
    assert "--last_per_updates" in finetune

    # NO invented output/save/ref flag -- ckpt dir derives from --dataset_name
    assert "--save_dir" not in finetune and "--output_dir" not in finetune
    assert "--ref_audio" not in finetune

    assert all("conda" in " ".join(c) and "f5tts" in " ".join(c) for c in calls)


def test_trainer_copies_checkpoint_and_vocab_into_out_dir(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    data_root, ckpts_root = _make_roots(tmp_path)
    calls = []
    out_dir = tmp_path / "out"
    trainer = F5Trainer(conda_env="f5tts", dataset_name="danil",
                        data_root=str(data_root), ckpts_root=str(ckpts_root),
                        runner=_fake_runner(calls, data_root, ckpts_root))
    result = trainer.train(str(manifest), str(out_dir),
                           {"epochs": 60, "learning_rate": 1e-5})
    # the real ckpt FILE + vocab were copied into out_dir so registry/serve finds them
    assert (out_dir / "model_last.pt").exists()
    assert (out_dir / "model_last.pt").read_bytes() == b"FAKE_CKPT"
    assert (out_dir / "vocab.txt").exists()
    assert result.checkpoint_dir == str(out_dir)
    assert result.steps == 60


def test_trainer_uses_default_hyperparams(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    data_root, ckpts_root = _make_roots(tmp_path)
    calls = []
    trainer = F5Trainer(conda_env="f5tts", dataset_name="danil",
                        data_root=str(data_root), ckpts_root=str(ckpts_root),
                        runner=_fake_runner(calls, data_root, ckpts_root))
    result = trainer.train(str(manifest), str(tmp_path / "out"), {})
    finetune = calls[1]
    assert "1e-05" in finetune or "1e-5" in finetune          # default LR 1e-5
    assert "3200" in finetune                                  # default frame batch
    assert "frame" in finetune
    assert "1" in finetune                                     # default grad_accumulation_steps
    assert result.steps == 60                                  # default epochs


def test_trainer_honors_non_default_config_overrides(tmp_path):
    """Override test MUST use values that differ from the defaults, so it actually proves
    the config flows through (defaults are lr=1e-5, bs=3200, epochs=60, grad_accum=1)."""
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    data_root, ckpts_root = _make_roots(tmp_path)
    calls = []
    trainer = F5Trainer(conda_env="f5tts", dataset_name="danil",
                        data_root=str(data_root), ckpts_root=str(ckpts_root),
                        runner=_fake_runner(calls, data_root, ckpts_root))
    result = trainer.train(str(manifest), str(tmp_path / "out"),
                           {"learning_rate": 2e-5, "batch_size_per_gpu": 1600,
                            "epochs": 42, "grad_accumulation_steps": 4})
    finetune = calls[1]
    flat = " ".join(finetune)
    # exact NON-DEFAULT values must appear (and the defaults must NOT)
    assert "2e-05" in finetune or "2e-5" in finetune
    assert "1e-05" not in finetune and "1e-5" not in finetune
    assert "1600" in finetune and "3200" not in finetune
    assert "42" in finetune
    # epochs flag carries the override, and result.steps echoes it
    assert "--epochs 42" in flat
    assert "--grad_accumulation_steps 4" in flat
    assert result.steps == 42


def test_trainer_raises_when_prepare_fails(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    data_root, ckpts_root = _make_roots(tmp_path)

    def fail_first(cmd, **kw):
        class R:
            returncode = 1
        return R()
    trainer = F5Trainer(conda_env="f5tts", dataset_name="danil",
                        data_root=str(data_root), ckpts_root=str(ckpts_root),
                        runner=fail_first)
    with pytest.raises(RuntimeError):
        trainer.train(str(manifest), str(tmp_path / "out"), {})


def test_trainer_raises_when_pretrained_vocab_missing(tmp_path):
    """finetune mode requires the pretrained Emilia_ZH_EN_pinyin/vocab.txt prereq under
    data_root; a clear error must name the expected path + the hf:// fetch source."""
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    data_root = tmp_path / "data"
    ckpts_root = tmp_path / "ckpts"
    data_root.mkdir(parents=True, exist_ok=True)
    ckpts_root.mkdir(parents=True, exist_ok=True)
    # NOTE: deliberately do NOT create Emilia_ZH_EN_pinyin/vocab.txt
    calls = []
    trainer = F5Trainer(conda_env="f5tts", dataset_name="danil",
                        data_root=str(data_root), ckpts_root=str(ckpts_root),
                        runner=_fake_runner(calls, data_root, ckpts_root))
    with pytest.raises(FileNotFoundError) as ei:
        trainer.train(str(manifest), str(tmp_path / "out"), {})
    msg = str(ei.value)
    assert "Emilia_ZH_EN_pinyin" in msg and "vocab.txt" in msg
    assert "hf://" in msg or "SWivid/F5-TTS" in msg


def test_trainer_raises_clear_error_when_checkpoint_missing(tmp_path):
    """If the trainer ran but no model_last.pt landed in <ckpts_root>/{name}, the copy
    step must raise a FileNotFoundError that names the expected checkpoint path."""
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    data_root, ckpts_root = _make_roots(tmp_path)
    calls = []

    def runner_no_ckpt(cmd, **kw):
        calls.append(cmd)
        flat = " ".join(cmd)
        if "prepare_csv_wavs" in flat:
            prepared = Path(data_root) / "danil_pinyin"
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / "vocab.txt").write_text("a\nb\n", encoding="utf-8")
        # finetune "succeeds" but writes NO checkpoint

        class R:
            returncode = 0
        return R()
    trainer = F5Trainer(conda_env="f5tts", dataset_name="danil",
                        data_root=str(data_root), ckpts_root=str(ckpts_root),
                        runner=runner_no_ckpt)
    with pytest.raises(FileNotFoundError) as ei:
        trainer.train(str(manifest), str(tmp_path / "out"), {})
    assert "model_last.pt" in str(ei.value)
