import pytest

from voxclone.prep.manifest import ClipRecord, write_manifest
from voxclone.train.xtts import manifest_to_xtts, XTTSTrainer
from voxclone.train.base import TrainResult


def _rec(path, text, dur=2.0):
    return ClipRecord(audio_path=path, text=text, duration=dur,
                      transcript_confidence=0.9, clipped_fraction=0.0,
                      category="harvard")


# --------------------------------------------------------------------------
# manifest_to_xtts converter (coqui-format CSVs)
# --------------------------------------------------------------------------

def test_emits_coqui_header_abs_paths_and_speaker(tmp_path):
    records = [_rec("/abs/a.wav", "hello there"), _rec("/abs/b.wav", "good morning")]
    train_csv, eval_csv = manifest_to_xtts(records, tmp_path, speaker="danil",
                                           eval_fraction=0.5, seed=0)
    train_lines = (tmp_path / "metadata_train.csv").read_text().splitlines()
    eval_lines = (tmp_path / "metadata_eval.csv").read_text().splitlines()
    # header row, pipe-delimited schema audio_file|text|speaker_name
    assert train_lines[0] == "audio_file|text|speaker_name"
    assert eval_lines[0] == "audio_file|text|speaker_name"
    body = [ln for ln in train_lines[1:] + eval_lines[1:] if ln]
    assert len(body) == 2
    for ln in body:
        cols = ln.split("|")
        assert cols[0].startswith("/abs/")     # abs path preserved (root_path="")
        assert cols[2] == "danil"
    assert str(train_csv).endswith("metadata_train.csv")
    assert str(eval_csv).endswith("metadata_eval.csv")


def test_skips_empty_text(tmp_path):
    records = [_rec("/abs/a.wav", "hi"), _rec("/abs/b.wav", "   ")]
    manifest_to_xtts(records, tmp_path, speaker="danil", eval_fraction=0.0, seed=0)
    body = [ln for ln in (tmp_path / "metadata_train.csv").read_text().splitlines()[1:] if ln]
    assert len(body) == 1


def test_filters_long_wav_and_long_text(tmp_path):
    long_text = "x " * 150  # > 200 chars
    records = [
        _rec("/abs/ok.wav", "fine", dur=2.0),
        _rec("/abs/longwav.wav", "fine", dur=20.0),    # > 11.6 s @ 22.05k -> drop
        _rec("/abs/longtext.wav", long_text, dur=2.0),  # > 200 chars -> drop
    ]
    manifest_to_xtts(records, tmp_path, speaker="danil", eval_fraction=0.0, seed=0)
    body = [ln for ln in (tmp_path / "metadata_train.csv").read_text().splitlines()[1:] if ln]
    assert len(body) == 1
    assert "/abs/ok.wav" in body[0]


def test_eval_split_deterministic_by_seed(tmp_path):
    records = [_rec(f"/abs/{i}.wav", f"text {i}") for i in range(10)]
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    manifest_to_xtts(records, out_a, speaker="danil", eval_fraction=0.3, seed=7)
    manifest_to_xtts(records, out_b, speaker="danil", eval_fraction=0.3, seed=7)
    eval_a = (out_a / "metadata_eval.csv").read_text()
    eval_b = (out_b / "metadata_eval.csv").read_text()
    assert eval_a == eval_b
    eval_body = [ln for ln in eval_a.splitlines()[1:] if ln]
    assert len(eval_body) == 3   # round(10 * 0.3)


def test_max_wav_length_and_max_text_length_configurable(tmp_path):
    # max_wav_length in SAMPLES (default 255995 @ 22050 Hz ~= 11.6 s).
    records = [
        _rec("/abs/short.wav", "keep", dur=1.0),
        _rec("/abs/medium.wav", "drop", dur=5.0),
    ]
    # tighten the wav-length budget to ~2 s of samples -> the 5 s clip is dropped
    manifest_to_xtts(records, tmp_path, speaker="danil", eval_fraction=0.0,
                     seed=0, max_wav_length=int(2.0 * 22050), max_text_length=200)
    body = [ln for ln in (tmp_path / "metadata_train.csv").read_text().splitlines()[1:] if ln]
    assert len(body) == 1
    assert "/abs/short.wav" in body[0]


# --------------------------------------------------------------------------
# XTTSTrainer (TrainAdapter, subprocess into xtts conda env, injected runner)
# --------------------------------------------------------------------------

def _write_manifest(p):
    write_manifest(p, [
        ClipRecord(audio_path="/abs/a.wav", text="hello", duration=2.0,
                   transcript_confidence=0.9, clipped_fraction=0.0, category="harvard"),
        ClipRecord(audio_path="/abs/b.wav", text="world", duration=2.0,
                   transcript_confidence=0.9, clipped_fraction=0.0, category="harvard"),
    ])


def test_trainer_emits_csvs_and_invokes_runner(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    calls = []

    def fake_runner(cmd, **kw):
        calls.append(cmd)

        class R:
            returncode = 0
        return R()

    trainer = XTTSTrainer(xtts_root="/home/prada/code/danill/third_party/xtts_v2_model",
                          conda_env="xtts", speaker="danil", runner=fake_runner)
    # Override with values that DIFFER from the trainer's own defaults
    # (default batch_size=4, grad_accum=2, lr=5e-6) so the test actually proves
    # config is honored — a trainer that ignored config would emit the defaults.
    result = trainer.train(str(manifest), str(tmp_path / "out"),
                           {"epochs": 8, "batch_size": 5, "grad_accum": 3, "lr": 1e-6})
    assert isinstance(result, TrainResult)
    assert result.checkpoint_dir == str(tmp_path / "out")
    assert result.steps == 8
    assert (tmp_path / "out" / "metadata_train.csv").exists()
    assert (tmp_path / "out" / "metadata_eval.csv").exists()
    assert len(calls) == 1
    cmd = calls[0]
    flat = " ".join(cmd)
    assert "conda" in flat and "xtts" in flat            # runs in the xtts conda env
    # the token AFTER each flag must be the exact non-default override value
    assert cmd[cmd.index("--epochs") + 1] == "8"
    assert cmd[cmd.index("--batch_size") + 1] == "5"
    assert cmd[cmd.index("--grad_accum") + 1] == "3"
    assert cmd[cmd.index("--lr") + 1] == "1e-06"


def test_trainer_small_dataset_override_defaults(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    calls = []

    def fake_runner(cmd, **kw):
        calls.append(cmd)

        class R:
            returncode = 0
        return R()

    trainer = XTTSTrainer(xtts_root="/x", conda_env="xtts", runner=fake_runner)
    result = trainer.train(str(manifest), str(tmp_path / "out"), {})
    # default small-dataset overrides (NOT recipe big-corpus defaults)
    cmd = calls[0]
    flat = " ".join(cmd)
    assert "--epochs" in flat and "10" in cmd          # default epochs 10 (in 6-15)
    assert "--batch_size" in flat and "4" in cmd        # default batch_size 4 (in 3-6)
    assert "--grad_accum" in flat and "2" in cmd        # default grad_accum 2 (in 1-4)
    assert "--lr" in flat and "5e-06" in flat           # default lr 5e-6
    assert "--max_wav_length" in flat and "255995" in cmd
    assert "--max_text_length" in flat and "200" in cmd
    # recipe invoked by FILE PATH (xtts env has no voxclone/pydantic), NOT `-m module`
    assert any(c.endswith("xtts_recipe.py") for c in cmd)
    assert "-m" not in cmd and "voxclone.train.xtts_recipe" not in cmd
    assert result.steps == 10


def test_trainer_raises_on_nonzero_returncode(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)

    def fail_runner(cmd, **kw):
        class R:
            returncode = 1
        return R()

    trainer = XTTSTrainer(xtts_root="/x", conda_env="xtts", runner=fail_runner)
    with pytest.raises(RuntimeError):
        trainer.train(str(manifest), str(tmp_path / "out"), {})
