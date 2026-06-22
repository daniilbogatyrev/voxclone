import numpy as np
import soundfile as sf

from voxclone.common.registry import ModelRegistry
from voxclone.eval.runner import run_eval


class FakeSynth:
    """Mirrors the SynthAdapter contract: bound checkpoint + synthesize(text, ref, params)."""

    def __init__(self, checkpoint, amp=1.0):
        self.checkpoint = checkpoint
        self._amp = amp

    def synthesize(self, text, reference_clip, params):
        return np.full(16000, self._amp, dtype=np.float32), 16000


def _write(path, sr=22050, seconds=2):
    sig = np.random.default_rng(0).normal(0, 0.1, sr * seconds).astype(np.float32)
    sf.write(str(path), sig, sr)


def _embedder(audio):
    assert isinstance(audio, np.ndarray)
    return np.array([1.0, 0.0])


def _transcriber(audio, sr):
    return [{"word": "hello", "probability": 1.0}, {"word": "world", "probability": 1.0}]


def _utmos(audio, sr):
    return 4.5


_WEIGHTS = {"similarity": 0.3, "naturalness": 0.5, "wer": 0.2}


def test_run_eval_persists_each_candidate_into_registry(tmp_path):
    p1, p2 = tmp_path / "real1.wav", tmp_path / "real2.wav"
    _write(p1); _write(p2)
    held_out = [("hello world", str(p1)), ("good morning", str(p2))]
    registry = ModelRegistry(tmp_path / "runs")

    result = run_eval(
        checkpoints={
            "xtts:finetuned": FakeSynth(checkpoint="runs/xtts"),
            "gptsovits:zeroshot": FakeSynth(checkpoint="v2Pro"),
        },
        held_out=held_out, reference_clip="ref.wav",
        embedder=_embedder, transcriber=_transcriber, utmos_model=_utmos,
        weights=_WEIGHTS, report_path=tmp_path / "report.md",
        registry=registry,
    )

    # Each candidate is registered under its own engine/model key with its checkpoint path.
    assert registry.best_checkpoint("xtts:finetuned") == "runs/xtts"
    assert registry.best_checkpoint("gptsovits:zeroshot") == "v2Pro"

    # The registered metrics carry the same 'score' best_checkpoint reads.
    raw = registry._load()
    assert raw["xtts:finetuned"][0]["metrics"]["score"] == result["metrics"]["xtts:finetuned"]["score"]
    assert "score" in raw["gptsovits:zeroshot"][0]["metrics"]


def test_best_checkpoint_round_trips_highest_score_under_one_key(tmp_path):
    p1, p2 = tmp_path / "real1.wav", tmp_path / "real2.wav"
    _write(p1); _write(p2)
    held_out = [("hello world", str(p1)), ("good morning", str(p2))]
    registry = ModelRegistry(tmp_path / "runs")

    # Two epochs of the SAME engine key; the louder/cleaner one (amp=1.0) scores higher
    # via similarity than the near-silent one (amp~0). Register the weaker one first.
    run_eval(
        checkpoints={"xtts": FakeSynth(checkpoint="runs/xtts/epoch3", amp=1e-4)},
        held_out=held_out, reference_clip="ref.wav",
        embedder=_embedder, transcriber=_transcriber, utmos_model=_utmos,
        weights=_WEIGHTS, report_path=tmp_path / "r1.md", registry=registry,
    )
    run_eval(
        checkpoints={"xtts": FakeSynth(checkpoint="runs/xtts/epoch10", amp=1.0)},
        held_out=held_out, reference_clip="ref.wav",
        embedder=_embedder, transcriber=_transcriber, utmos_model=_utmos,
        weights=_WEIGHTS, report_path=tmp_path / "r2.md", registry=registry,
    )

    entries = registry._load()["xtts"]
    assert len(entries) == 2  # both checkpoints persisted under the one key
    best = max(entries, key=lambda e: e["metrics"]["score"])
    assert registry.best_checkpoint("xtts") == best["checkpoint"]
    # best_checkpoint no longer returns None now that run_eval populated the registry.
    assert registry.best_checkpoint("xtts") is not None


def test_run_eval_without_registry_is_unchanged(tmp_path):
    p1, p2 = tmp_path / "real1.wav", tmp_path / "real2.wav"
    _write(p1); _write(p2)
    held_out = [("hello world", str(p1)), ("good morning", str(p2))]

    result = run_eval(
        checkpoints={"ckpt_10": FakeSynth(checkpoint="runs/x")},
        held_out=held_out, reference_clip="ref.wav",
        embedder=_embedder, transcriber=_transcriber, utmos_model=_utmos,
        weights=_WEIGHTS, report_path=tmp_path / "report.md",
    )
    # No registry passed -> existing return contract intact, no persistence side effect.
    assert result["best"] == "ckpt_10"
    assert "score" in result["metrics"]["ckpt_10"]
    assert not (tmp_path / "runs" / "registry.json").exists()
