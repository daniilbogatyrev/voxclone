import numpy as np
import soundfile as sf
from voxclone.common.registry import ModelRegistry
from voxclone.eval.runner import run_eval

class FakeSynth:
    def synthesize(self, text, reference_clip, params):
        return np.ones(16000, dtype=np.float32), 16000

def _write(path, sr=22050, seconds=2):
    sig = np.random.default_rng(0).normal(0, 0.1, sr * seconds).astype(np.float32)
    sf.write(str(path), sig, sr)

def test_registry_plus_eval_pipeline(tmp_path):
    p1 = tmp_path / "r1.wav"
    _write(p1)
    reg = ModelRegistry(tmp_path / "runs")
    result = run_eval(
        checkpoints={"ckpt_10": FakeSynth()},
        held_out=[("hello world", str(p1))],
        reference_clip="ref.wav",
        embedder=lambda a: np.array([1.0, 0.0]),
        transcriber=type("T", (), {"transcribe": lambda self, a, s:
            [{"word": "hello", "probability": 1.0}, {"word": "world", "probability": 1.0}]})(),
        utmos_model=lambda a, s: 4.5,
        weights={"similarity": 0.5, "naturalness": 0.3, "wer": 0.2},
        report_path=tmp_path / "reports/eval.md",
    )
    reg.register("gptsovits", "ckpt_10", {"score": result["metrics"][result["best"]]["score"]})
    assert reg.best_checkpoint("gptsovits") == "ckpt_10"
