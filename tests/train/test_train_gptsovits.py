"""Unit tests for the REAL GPT-SoVITS v2Pro prep + config-file fine-tune pipeline.

These tests are GPU-free: the subprocess runner is injected/mocked, and the config
files the trainer generates are written to tmp dirs and their contents asserted.

Ground truth (verified against the pinned repo third_party/GPT-SoVITS):
  - webui.py open1a  -> 1-get-text.py        (env-driven, NOT flags)
  - webui.py open1b  -> 2-get-hubert-wav32k.py  then  2-get-sv.py (v2Pro-mandatory)
  - webui.py open1c  -> 3-get-semantic.py
  - webui.py open1Ba -> s2_train.py  --config <tmp_s2.json>   (SoVITS)
  - webui.py open1Bb -> s1_train.py  --config_file <tmp_s1.yaml>  (GPT)
The crashing --list/--exp/--epochs flag stub must be GONE.
"""

import json
from pathlib import Path

import pytest
import yaml

from voxclone.prep.manifest import ClipRecord, write_manifest
from voxclone.train.base import TrainResult
from voxclone.train.gptsovits import GPTSoVITSTrainer, manifest_to_gptsovits


ROOT = "/opt/GPT-SoVITS"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rec(path, text, cat="harvard"):
    return ClipRecord(audio_path=path, text=text, duration=2.0,
                      transcript_confidence=0.9, clipped_fraction=0.0, category=cat)


def _write_manifest(p):
    write_manifest(p, [
        _rec("/data/a.wav", "hello world"),
        _rec("/data/b.wav", "good morning"),
    ])


class _Recorder:
    """Records (cmd, env) for each runner call; simulates the prep stages writing the
    intermediate artifacts a real run would produce, and the train stages writing weights.
    Returns success. cwd is whatever the trainer passes (should be the GPT-SoVITS root)."""

    def __init__(self, root):
        self.calls = []         # list of cmd lists
        self.envs = []          # list of env dicts (the env each cmd ran with)
        self.cwds = []          # list of cwd values
        self.root = Path(root)

    def __call__(self, cmd, env=None, cwd=None, **kw):
        self.calls.append(list(cmd))
        self.envs.append(dict(env) if env is not None else None)
        self.cwds.append(cwd)
        flat = " ".join(cmd)

        # base dir the prep scripts read from env opt_dir; create it so artifacts can land
        opt_dir = None
        if env is not None and env.get("opt_dir"):
            opt_dir = Path(env["opt_dir"])
            opt_dir.mkdir(parents=True, exist_ok=True)

        if "1-get-text.py" in flat and opt_dir is not None:
            (opt_dir / "2-name2text.txt").write_text("a\tphones\n", encoding="utf-8")
        elif "3-get-semantic.py" in flat and opt_dir is not None:
            (opt_dir / "6-name2semantic.tsv").write_text(
                "item_name\tsemantic_audio\n", encoding="utf-8")
        elif "s2_train.py" in flat:
            # SoVITS weights land in SoVITS_weights_v2Pro/
            wdir = self.root / "SoVITS_weights_v2Pro"
            wdir.mkdir(parents=True, exist_ok=True)
            (wdir / "danil_e8_s120.pth").write_bytes(b"FAKE_S2")
        elif "s1_train.py" in flat:
            wdir = self.root / "GPT_weights_v2Pro"
            wdir.mkdir(parents=True, exist_ok=True)
            (wdir / "danil-e15.ckpt").write_bytes(b"FAKE_S1")

        class R:
            returncode = 0
        return R()


def _run(tmp_path, exp_name="danil", config=None, root=None):
    """Drive a full train() with a recording runner; return (trainer, rec, result)."""
    root = root or str(tmp_path / "gsv_root")
    Path(root).mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    rec = _Recorder(root)
    trainer = GPTSoVITSTrainer(gptsovits_root=root, speaker="target",
                               exp_name=exp_name, runner=rec)
    result = trainer.train(str(manifest), str(tmp_path / "out"), config or {})
    return trainer, rec, result


# --------------------------------------------------------------------------- #
# manifest_to_gptsovits unchanged contract
# --------------------------------------------------------------------------- #
def test_manifest_converter_still_pipe_format():
    out = manifest_to_gptsovits([_rec("data/a.wav", "Hello world.")], speaker="target")
    assert out.strip() == "data/a.wav|target|EN|Hello world."


# --------------------------------------------------------------------------- #
# the old crashing flag stub must be gone
# --------------------------------------------------------------------------- #
def test_no_legacy_flag_stub_anywhere(tmp_path):
    _, rec, _ = _run(tmp_path)
    for cmd in rec.calls:
        flat = " ".join(cmd)
        assert "--list" not in flat, "legacy --list flag must be gone"
        assert "--exp" not in flat, "legacy --exp flag must be gone"
        assert "--epochs" not in flat, "legacy --epochs flag must be gone"


# --------------------------------------------------------------------------- #
# pipeline shape: .list + 4 prep stages + 2 train stages, IN ORDER
# --------------------------------------------------------------------------- #
def test_writes_dot_list_in_pipe_spk_en_text_format(tmp_path):
    _, rec, _ = _run(tmp_path)
    # the .list is the inp_text the prep stages read; it lives under opt_dir
    opt_dir = Path(rec.envs[0]["opt_dir"])
    inp_text = Path(rec.envs[0]["inp_text"])
    assert inp_text.exists()
    lines = inp_text.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "/data/a.wav|target|EN|hello world"
    assert lines[1] == "/data/b.wav|target|EN|good morning"
    assert opt_dir.name == "danil"


def test_six_stages_in_exact_order(tmp_path):
    _, rec, _ = _run(tmp_path)
    scripts = []
    for cmd in rec.calls:
        flat = " ".join(cmd)
        for token in ("1-get-text.py", "2-get-hubert-wav32k.py", "2-get-sv.py",
                      "3-get-semantic.py", "s2_train.py", "s1_train.py"):
            if token in flat:
                scripts.append(token)
                break
    assert scripts == [
        "1-get-text.py",
        "2-get-hubert-wav32k.py",
        "2-get-sv.py",            # v2Pro-mandatory speaker-verification embeddings
        "3-get-semantic.py",
        "s2_train.py",
        "s1_train.py",
    ]


def test_runner_invoked_with_root_as_cwd(tmp_path):
    root = str(tmp_path / "gsv_root")
    _, rec, _ = _run(tmp_path, root=root)
    # every stage runs with cwd == GPT-SoVITS root (webui runs from the repo root so the
    # relative GPT_SoVITS/... config + model paths resolve)
    assert all(c == root for c in rec.cwds)


# --------------------------------------------------------------------------- #
# stage 1a env: 1-get-text.py
# --------------------------------------------------------------------------- #
def test_stage_1a_env_has_inp_text_exp_bert_version(tmp_path):
    _, rec, _ = _run(tmp_path)
    cmd, env = rec.calls[0], rec.envs[0]
    assert "1-get-text.py" in " ".join(cmd)
    assert env["exp_name"] == "danil"
    assert env["opt_dir"].endswith("logs/danil")            # exp_root="logs"
    assert env["bert_pretrained_dir"] == \
        "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    assert env["version"] == "v2Pro"                         # 1-get-text reads version
    assert env["i_part"] == "0" and env["all_parts"] == "1"


# --------------------------------------------------------------------------- #
# stage 1b env: 2-get-hubert-wav32k.py + 2-get-sv.py
# --------------------------------------------------------------------------- #
def test_stage_hubert_env_has_cnhubert_and_sv_path(tmp_path):
    _, rec, _ = _run(tmp_path)
    cmd, env = rec.calls[1], rec.envs[1]
    assert "2-get-hubert-wav32k.py" in " ".join(cmd)
    assert env["cnhubert_base_dir"] == \
        "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    assert env["sv_path"] == \
        "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
    assert env["opt_dir"].endswith("logs/danil")


def test_stage_sv_env_has_sv_path(tmp_path):
    _, rec, _ = _run(tmp_path)
    cmd, env = rec.calls[2], rec.envs[2]
    assert "2-get-sv.py" in " ".join(cmd)
    assert env["sv_path"] == \
        "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"


# --------------------------------------------------------------------------- #
# stage 1c env: 3-get-semantic.py
# --------------------------------------------------------------------------- #
def test_stage_semantic_env_has_pretrained_s2g_and_s2config(tmp_path):
    _, rec, _ = _run(tmp_path)
    cmd, env = rec.calls[3], rec.envs[3]
    assert "3-get-semantic.py" in " ".join(cmd)
    assert env["pretrained_s2G"] == \
        "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
    assert env["s2config_path"] == "GPT_SoVITS/configs/s2v2Pro.json"


# --------------------------------------------------------------------------- #
# s2 (SoVITS) training: tmp_s2.json + entrypoint
# --------------------------------------------------------------------------- #
def test_s2_train_uses_config_flag_not_legacy_flags(tmp_path):
    _, rec, _ = _run(tmp_path)
    cmd = rec.calls[4]
    flat = " ".join(cmd)
    assert "s2_train.py" in flat
    assert "--config" in cmd                                 # config-file driven
    # the config arg is the tmp_s2.json the trainer wrote
    cfg_path = Path(cmd[cmd.index("--config") + 1])
    assert cfg_path.name == "tmp_s2.json"
    assert cfg_path.exists()


def test_s2_json_injects_v2pro_pretrained_weights_and_train_fields(tmp_path):
    _, rec, _ = _run(tmp_path, config={"epochs_sovits": 8, "batch_size": 12})
    cmd = rec.calls[4]
    cfg = json.loads(Path(cmd[cmd.index("--config") + 1]).read_text(encoding="utf-8"))
    assert cfg["model"]["version"] == "v2Pro"
    assert cfg["version"] == "v2Pro"
    assert cfg["train"]["epochs"] == 8
    assert cfg["train"]["batch_size"] == 12
    assert cfg["train"]["pretrained_s2G"] == \
        "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
    assert cfg["train"]["pretrained_s2D"] == \
        "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth"   # s2G -> s2D
    assert cfg["save_weight_dir"] == "SoVITS_weights_v2Pro"
    assert cfg["name"] == "danil"
    # data exp_dir + s2_ckpt_dir point at logs/<exp>
    assert cfg["data"]["exp_dir"].endswith("logs/danil")
    assert cfg["s2_ckpt_dir"].endswith("logs/danil")


def test_s2_json_inherits_base_config_defaults(tmp_path):
    """Fields NOT overridden must come straight from configs/s2v2Pro.json (e.g.
    text_low_lr_rate=0.4, sampling_rate=32000)."""
    _, rec, _ = _run(tmp_path)
    cmd = rec.calls[4]
    cfg = json.loads(Path(cmd[cmd.index("--config") + 1]).read_text(encoding="utf-8"))
    assert cfg["train"]["text_low_lr_rate"] == 0.4
    assert cfg["data"]["sampling_rate"] == 32000


# --------------------------------------------------------------------------- #
# s1 (GPT) training: tmp_s1.yaml + entrypoint
# --------------------------------------------------------------------------- #
def test_s1_train_uses_config_file_flag(tmp_path):
    _, rec, _ = _run(tmp_path)
    cmd = rec.calls[5]
    flat = " ".join(cmd)
    assert "s1_train.py" in flat
    assert "--config_file" in cmd                            # s1 entrypoint flag
    cfg_path = Path(cmd[cmd.index("--config_file") + 1])
    assert cfg_path.name == "tmp_s1.yaml"
    assert cfg_path.exists()


def test_s1_yaml_injects_pretrained_paths_and_train_fields(tmp_path):
    _, rec, _ = _run(tmp_path, config={"epochs_gpt": 15})
    cmd = rec.calls[5]
    cfg = yaml.safe_load(Path(cmd[cmd.index("--config_file") + 1]).read_text())
    assert cfg["pretrained_s1"] == "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    assert cfg["train"]["epochs"] == 15
    assert cfg["train"]["exp_name"] == "danil"
    assert cfg["train"]["half_weights_save_dir"] == "GPT_weights_v2Pro"
    assert cfg["train_semantic_path"].endswith("logs/danil/6-name2semantic.tsv")
    assert cfg["train_phoneme_path"].endswith("logs/danil/2-name2text.txt")
    assert cfg["output_dir"].endswith("logs/danil/logs_s1_v2Pro")


def test_s1_yaml_inherits_base_defaults(tmp_path):
    """Non-overridden fields come from configs/s1longer-v2.yaml (e.g. data.max_sec=54)."""
    _, rec, _ = _run(tmp_path)
    cmd = rec.calls[5]
    cfg = yaml.safe_load(Path(cmd[cmd.index("--config_file") + 1]).read_text())
    assert cfg["data"]["max_sec"] == 54
    assert cfg["model"]["EOS"] == 1024


def test_s1_train_sets_hz_env_25hz(tmp_path):
    _, rec, _ = _run(tmp_path)
    env = rec.envs[5]
    assert env["hz"] == "25hz"                               # os.environ['hz']='25hz'


# --------------------------------------------------------------------------- #
# defaults: epochs ~8 / ~15 per the research doc + webui caps
# --------------------------------------------------------------------------- #
def test_default_epochs_sovits_8_gpt_15(tmp_path):
    _, rec, _ = _run(tmp_path, config={})
    s2 = json.loads(Path(rec.calls[4][rec.calls[4].index("--config") + 1]).read_text())
    s1 = yaml.safe_load(Path(rec.calls[5][rec.calls[5].index("--config_file") + 1]).read_text())
    assert s2["train"]["epochs"] == 8
    assert s1["train"]["epochs"] == 15


# --------------------------------------------------------------------------- #
# result points at the v2Pro weight dirs
# --------------------------------------------------------------------------- #
def test_result_points_at_v2pro_weight_dirs(tmp_path):
    _, rec, result = _run(tmp_path, config={"epochs_sovits": 8, "epochs_gpt": 15})
    assert isinstance(result, TrainResult)
    assert "SoVITS_weights_v2Pro" in result.checkpoint_dir
    assert result.steps == 8 + 15


# --------------------------------------------------------------------------- #
# failure propagation
# --------------------------------------------------------------------------- #
def test_raises_when_a_prep_stage_fails(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    root = tmp_path / "gsv_root"
    root.mkdir(parents=True, exist_ok=True)

    def fail_first(cmd, env=None, cwd=None, **kw):
        class R:
            returncode = 1
        return R()

    trainer = GPTSoVITSTrainer(gptsovits_root=str(root), runner=fail_first)
    with pytest.raises(RuntimeError):
        trainer.train(str(manifest), str(tmp_path / "out"), {})
