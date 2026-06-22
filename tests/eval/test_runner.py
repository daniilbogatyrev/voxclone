import pytest
import numpy as np
import soundfile as sf
from voxclone.eval.runner import run_eval

class FakeSynth:
    def synthesize(self, text, reference_clip, params):
        return np.ones(16000, dtype=np.float32), 16000

def _write(path, sr=22050, seconds=2):
    sig = np.random.default_rng(0).normal(0, 0.1, sr * seconds).astype(np.float32)
    sf.write(str(path), sig, sr)

def test_run_eval_returns_metrics_and_best(tmp_path):
    p1, p2 = tmp_path / "real1.wav", tmp_path / "real2.wav"
    _write(p1); _write(p2)
    held_out = [("hello world", str(p1)), ("good morning", str(p2))]
    def embedder(audio):                      # now receives a 16 kHz mono array
        assert isinstance(audio, np.ndarray)
        return np.array([1.0, 0.0])
    def transcriber(audio, sr):
        return [{"word": "hello", "probability": 1.0}, {"word": "world", "probability": 1.0}]
    utmos = lambda audio, sr: 4.5
    result = run_eval(
        checkpoints={"ckpt_10": FakeSynth()}, held_out=held_out, reference_clip="ref.wav",
        embedder=embedder, transcriber=transcriber, utmos_model=utmos,
        weights={"similarity": 0.3, "naturalness": 0.5, "wer": 0.2},
        report_path=tmp_path / "report.md", wer_dq_threshold=0.2,
    )
    assert result["best"] == "ckpt_10"
    assert 0.0 <= result["metrics"]["ckpt_10"]["similarity"] <= 1.0
    assert "ceiling" in result
    assert (tmp_path / "report.md").exists()

def test_run_eval_rejects_empty_inputs(tmp_path):
    with pytest.raises(ValueError):
        run_eval(checkpoints={}, held_out=[("hi", "r.wav")], reference_clip="ref.wav",
                 embedder=lambda a: np.array([1.0, 0.0]),
                 transcriber=lambda a, s: [{"word": "hi", "probability": 1.0}],
                 utmos_model=lambda a, s: 4.0,
                 weights={"similarity": 0.3, "naturalness": 0.5, "wer": 0.2},
                 report_path=tmp_path / "r.md")
