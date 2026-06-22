from pathlib import Path
import pytest
from voxclone.common.config import load_config, AppConfig

def test_load_default_config(tmp_path: Path):
    cfg_text = (
        "capture:\n  sample_rate: 48000\n  target_minutes: 50\n"
        "prep:\n  target_sample_rate: 48000\n  vad_min_duration_s: 3.0\n"
        "  vad_max_duration_s: 15.0\n  denoise_enabled: true\n  peak_dbfs: -1.0\n"
        "  whisper_model: large-v3\n  whisper_device: cuda\n"
        "  validation:\n    min_duration_s: 1.5\n    max_duration_s: 15.0\n"
        "    max_clipped_fraction: 0.005\n    min_transcript_confidence: 0.6\n"
    )
    p = tmp_path / "c.yaml"
    p.write_text(cfg_text)
    cfg = load_config(p)
    assert isinstance(cfg, AppConfig)
    assert cfg.prep.vad_max_duration_s == 15.0
    assert cfg.prep.validation.min_transcript_confidence == 0.6

def test_invalid_config_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("prep:\n  vad_min_duration_s: not_a_number\n")
    with pytest.raises(Exception):
        load_config(p)

def test_eval_config_defaults():
    from voxclone.common.config import AppConfig, EvalConfig
    cfg = AppConfig()
    assert isinstance(cfg.eval, EvalConfig)
    assert cfg.eval.similarity == 0.30
    assert cfg.eval.naturalness == 0.50
    assert cfg.eval.wer == 0.20
    assert cfg.eval.wer_dq_threshold == 0.20
    assert cfg.eval.target_lufs == -23.0
    assert cfg.eval.eval_sr == 16000

def test_eval_config_weights_dict_helper():
    from voxclone.common.config import EvalConfig
    assert EvalConfig().weights() == {"similarity": 0.30, "naturalness": 0.50, "wer": 0.20}
