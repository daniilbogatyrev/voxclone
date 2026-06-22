"""GPU-free argparse/dispatch tests for scripts/train.py multi-engine wiring.

scripts.train selects the Trainer class from voxclone.train.TRAIN_ENGINES per
``--engine`` and calls ``.train(manifest, out, config)``. Heavy trainers are never
constructed: we monkeypatch TRAIN_ENGINES with recording fakes, so no torch / TTS /
conda / GPU is touched. This pins the naming contract (``--out`` defaults to
``runs/<engine>`` whose basename == the registry key == the engine name).
"""
import pytest

import scripts.train as train


# --------------------------------------------------------------------------
# A recording fake that mirrors the TrainAdapter contract without heavy deps.
# --------------------------------------------------------------------------
class FakeTrainer:
    instances: list["FakeTrainer"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.train_calls: list[tuple] = []
        # record into THIS class's own instances list (per-engine subclass)
        type(self).instances.append(self)

    def train(self, manifest_path, out_dir, config):
        self.train_calls.append((manifest_path, out_dir, config))

        class R:
            checkpoint_dir = out_dir
            steps = 0
        return R()


def _patch_engines(monkeypatch, *engines):
    """Replace each named engine in TRAIN_ENGINES with a distinct recording fake."""
    FakeTrainer.instances = []
    fakes = {}
    for eng in engines:
        cls = type(f"Fake_{eng}", (FakeTrainer,), {"instances": []})
        fakes[eng] = cls
        monkeypatch.setitem(train.TRAIN_ENGINES, eng, cls)
    return fakes


def _cfg(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  epochs: 7\n")
    return str(p)


# --------------------------------------------------------------------------
# Engine selection / dispatch through TRAIN_ENGINES
# --------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["xtts", "f5", "gptsovits"])
def test_dispatches_to_engine_trainer_from_train_map(monkeypatch, tmp_path, engine):
    fakes = _patch_engines(monkeypatch, engine)
    argv = ["--engine", engine, "--manifest", str(tmp_path / "m.jsonl"),
            "--config", _cfg(tmp_path), "--out", str(tmp_path / "out")]
    if engine == "gptsovits":
        argv += ["--gptsovits-root", "/opt/GPT-SoVITS"]
    train.main(argv)
    # Exactly the selected engine's class was constructed and trained.
    cls = fakes[engine]
    assert len(cls.instances) == 1, f"{engine} trainer not constructed exactly once"
    inst = cls.instances[0]
    assert len(inst.train_calls) == 1


def test_only_the_selected_engine_is_constructed(monkeypatch, tmp_path):
    fakes = _patch_engines(monkeypatch, "xtts", "f5", "gptsovits")
    train.main(["--engine", "f5", "--manifest", str(tmp_path / "m.jsonl"),
                "--config", _cfg(tmp_path), "--out", str(tmp_path / "out")])
    assert len(fakes["f5"].instances) == 1
    assert fakes["xtts"].instances == []        # other engines untouched
    assert fakes["gptsovits"].instances == []


def test_train_called_with_parsed_config_dict(monkeypatch, tmp_path):
    fakes = _patch_engines(monkeypatch, "xtts")
    train.main(["--engine", "xtts", "--manifest", str(tmp_path / "m.jsonl"),
                "--config", _cfg(tmp_path), "--out", str(tmp_path / "out")])
    manifest_path, out_dir, config = fakes["xtts"].instances[0].train_calls[0]
    assert manifest_path == str(tmp_path / "m.jsonl")
    assert out_dir == str(tmp_path / "out")
    # the parsed config dict is the YAML's `train:` sub-mapping
    assert config == {"epochs": 7}


# --------------------------------------------------------------------------
# Naming contract: --out defaults to runs/<engine>
# --------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["xtts", "f5", "gptsovits"])
def test_out_defaults_to_runs_engine(monkeypatch, tmp_path, engine):
    fakes = _patch_engines(monkeypatch, engine)
    argv = ["--engine", engine, "--manifest", str(tmp_path / "m.jsonl"),
            "--config", _cfg(tmp_path)]
    if engine == "gptsovits":
        argv += ["--gptsovits-root", "/opt/GPT-SoVITS"]
    train.main(argv)
    _, out_dir, _ = fakes[engine].instances[0].train_calls[0]
    # basename of --out == engine key == registry key == serve model name
    assert out_dir == f"runs/{engine}"
    assert out_dir.rsplit("/", 1)[-1] == engine


def test_explicit_out_overrides_default(monkeypatch, tmp_path):
    fakes = _patch_engines(monkeypatch, "xtts")
    train.main(["--engine", "xtts", "--manifest", str(tmp_path / "m.jsonl"),
                "--config", _cfg(tmp_path), "--out", "runs/custom_xtts"])
    _, out_dir, _ = fakes["xtts"].instances[0].train_calls[0]
    assert out_dir == "runs/custom_xtts"


# --------------------------------------------------------------------------
# Conditional roots: --gptsovits-root only for gptsovits; --xtts-root for xtts
# --------------------------------------------------------------------------

def test_gptsovits_root_required_only_for_gptsovits(monkeypatch, tmp_path):
    _patch_engines(monkeypatch, "gptsovits")
    # gptsovits with no --gptsovits-root must error (the root is mandatory for it).
    with pytest.raises(SystemExit):
        train.main(["--engine", "gptsovits", "--manifest", str(tmp_path / "m.jsonl"),
                    "--config", _cfg(tmp_path)])


def test_xtts_and_f5_do_not_require_gptsovits_root(monkeypatch, tmp_path):
    fakes = _patch_engines(monkeypatch, "xtts", "f5")
    for engine in ("xtts", "f5"):
        train.main(["--engine", engine, "--manifest", str(tmp_path / "m.jsonl"),
                    "--config", _cfg(tmp_path)])
        assert len(fakes[engine].instances) == 1   # built without --gptsovits-root


def test_gptsovits_root_passed_to_trainer_ctor(monkeypatch, tmp_path):
    fakes = _patch_engines(monkeypatch, "gptsovits")
    train.main(["--engine", "gptsovits", "--manifest", str(tmp_path / "m.jsonl"),
                "--config", _cfg(tmp_path), "--gptsovits-root", "/opt/GPT-SoVITS"])
    ctor_kwargs = fakes["gptsovits"].instances[0].kwargs
    assert ctor_kwargs.get("gptsovits_root") == "/opt/GPT-SoVITS"


def test_xtts_root_passed_to_trainer_ctor(monkeypatch, tmp_path):
    fakes = _patch_engines(monkeypatch, "xtts")
    train.main(["--engine", "xtts", "--manifest", str(tmp_path / "m.jsonl"),
                "--config", _cfg(tmp_path), "--xtts-root", "/models/xtts_v2"])
    ctor_kwargs = fakes["xtts"].instances[0].kwargs
    assert ctor_kwargs.get("xtts_root") == "/models/xtts_v2"


# --------------------------------------------------------------------------
# Unknown engine is rejected by argparse (choices restricted to TRAIN_ENGINES).
# --------------------------------------------------------------------------

def test_unknown_engine_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        train.main(["--engine", "bogus", "--manifest", str(tmp_path / "m.jsonl"),
                    "--config", _cfg(tmp_path)])
