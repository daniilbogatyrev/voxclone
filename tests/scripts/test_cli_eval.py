"""GPU-free argparse/dispatch tests for scripts/eval.py multi-engine wiring.

scripts.eval parses each ``--candidates`` token as ``engine:label=checkpoint`` and
builds the candidate via ``voxclone.synth.SYNTH_ENGINES[engine](checkpoint=path)`` --
NOT a hardcoded GPTSoVITSSynth. Weights come from ``EvalConfig.weights()`` (0.30 /
0.50 / 0.20) and ``EvalConfig.wer_dq_threshold`` (0.20) is threaded through, replacing
the old hardcoded 0.5/0.3/0.2 dict. After scoring, each candidate is registered into a
``ModelRegistry`` keyed by engine so ``best_checkpoint`` resolves.

The whole thing is GPU-free: SYNTH_ENGINES classes are monkeypatched with recording
fakes, the model loaders (load_ecapa / load_utmos / load_transcriber) are replaced with
trivial stand-ins, and the registry is a tmp dir. No torch / TTS / f5_tts / chatterbox /
whisper is imported, and no server / training run is launched.
"""
import numpy as np
import soundfile as sf
import pytest

import scripts.eval as cli_eval
from voxclone.common.registry import ModelRegistry
from voxclone.eval.runner import run_eval


# --------------------------------------------------------------------------
# Recording fake mirroring the SynthAdapter contract (bound checkpoint +
# synthesize(text, ref, params)). Each engine gets its own subclass so we can
# assert which class each candidate token resolved to.
# --------------------------------------------------------------------------
class FakeSynth:
    instances: list["FakeSynth"] = []

    def __init__(self, checkpoint, **kwargs):
        self.checkpoint = checkpoint
        self.kwargs = kwargs
        type(self).instances.append(self)

    def synthesize(self, text, reference_clip, params):
        # Deterministic, non-silent 16 kHz clip so similarity/utmos are well-defined.
        return np.full(16000, 0.5, dtype=np.float32), 16000


def _patch_engines(monkeypatch, *engines):
    """Replace each named engine in SYNTH_ENGINES with a distinct recording fake."""
    FakeSynth.instances = []
    fakes = {}
    for eng in engines:
        cls = type(f"Fake_{eng}", (FakeSynth,), {"instances": []})
        fakes[eng] = cls
        monkeypatch.setitem(cli_eval.SYNTH_ENGINES, eng, cls)
    return fakes


def _embedder(audio):
    assert isinstance(audio, np.ndarray)
    return np.array([1.0, 0.0])


def _transcriber(audio, sr):
    return [{"word": "hello", "probability": 1.0}, {"word": "world", "probability": 1.0}]


def _utmos(audio, sr):
    return 4.5


def _patch_loaders(monkeypatch, utmos_calls=None):
    """Replace the heavy model loaders with GPU-free stand-ins.

    The UTMOS loader patched is ``load_speechmos_utmos`` -- the working torch.hub
    SpeechMOS loader -- NOT the uninstalled-utmosv2 ``load_utmos``; pass a list as
    ``utmos_calls`` to record that it was the one invoked (no real torch.hub download).
    """
    monkeypatch.setattr(cli_eval, "load_ecapa", lambda *a, **k: _embedder)

    def _fake_speechmos(*a, **k):
        if utmos_calls is not None:
            utmos_calls.append((a, k))
        return _utmos

    monkeypatch.setattr(cli_eval, "load_speechmos_utmos", _fake_speechmos)
    monkeypatch.setattr(cli_eval, "load_transcriber", lambda *a, **k: _transcriber)


def _heldout(tmp_path):
    p1, p2 = tmp_path / "real1.wav", tmp_path / "real2.wav"
    for p in (p1, p2):
        sig = np.random.default_rng(0).normal(0, 0.1, 22050 * 2).astype(np.float32)
        sf.write(str(p), sig, 22050)
    tsv = tmp_path / "held.tsv"
    tsv.write_text(f"hello world\t{p1}\ngood morning\t{p2}\n")
    return str(tsv)


def _run(monkeypatch, tmp_path, *engines, candidates, extra_argv=None):
    fakes = _patch_engines(monkeypatch, *engines)
    _patch_loaders(monkeypatch)
    held = _heldout(tmp_path)
    runs = tmp_path / "runs"
    argv = ["--candidates", *candidates,
            "--reference", str(tmp_path / "ref.wav"),
            "--held-out", held,
            "--report", str(tmp_path / "report.md"),
            "--register", str(runs),
            "--device", "cpu"]
    if extra_argv:
        argv += extra_argv
    cli_eval.main(argv)
    return fakes, runs


# --------------------------------------------------------------------------
# Candidate parsing: engine:label=checkpoint -> SYNTH_ENGINES[engine](checkpoint=...)
# --------------------------------------------------------------------------

def test_candidate_built_via_synth_engines_not_hardcoded_gptsovits(monkeypatch, tmp_path):
    fakes, _ = _run(monkeypatch, tmp_path, "xtts",
                    candidates=["xtts:finetuned=runs/xtts"])
    # The xtts candidate was built from SYNTH_ENGINES["xtts"], not GPTSoVITSSynth.
    assert len(fakes["xtts"].instances) == 1
    assert fakes["xtts"].instances[0].checkpoint == "runs/xtts"


def test_each_engine_resolves_to_its_own_synth_class(monkeypatch, tmp_path):
    fakes, _ = _run(
        monkeypatch, tmp_path, "xtts", "f5", "gptsovits", "chatterbox",
        candidates=["xtts:finetuned=runs/xtts", "f5:finetuned=runs/f5",
                    "gptsovits:finetuned=runs/gptsovits", "gptsovits:zeroshot=v2Pro",
                    "chatterbox:zeroshot=base"])
    # Exactly the right class per engine token; gptsovits builds twice (finetuned+zeroshot).
    assert len(fakes["xtts"].instances) == 1
    assert len(fakes["f5"].instances) == 1
    assert len(fakes["gptsovits"].instances) == 2
    assert len(fakes["chatterbox"].instances) == 1
    # Checkpoints flow from the =path half of each token.
    assert {i.checkpoint for i in fakes["gptsovits"].instances} == {"runs/gptsovits", "v2Pro"}
    assert fakes["chatterbox"].instances[0].checkpoint == "base"


def test_five_candidates_build_correct_classes_and_populate_registry(monkeypatch, tmp_path):
    candidates = ["xtts:finetuned=runs/xtts", "f5:finetuned=runs/f5",
                  "gptsovits:finetuned=runs/gptsovits", "gptsovits:zeroshot=v2Pro",
                  "chatterbox:zeroshot=base"]
    fakes, runs = _run(monkeypatch, tmp_path, "xtts", "f5", "gptsovits", "chatterbox",
                       candidates=candidates)
    # All five candidate classes were constructed.
    total = sum(len(fakes[e].instances) for e in ("xtts", "f5", "gptsovits", "chatterbox"))
    assert total == 5
    # Registry is populated: every candidate key resolves to its bound checkpoint.
    reg = ModelRegistry(runs)
    raw = reg._load()
    assert set(raw) == {"xtts_finetuned", "f5_finetuned", "gptsovits_finetuned",
                        "gptsovits_zeroshot", "chatterbox_zeroshot"}
    assert reg.best_checkpoint("xtts_finetuned") == "runs/xtts"
    assert reg.best_checkpoint("gptsovits_zeroshot") == "v2Pro"
    assert reg.best_checkpoint("chatterbox_zeroshot") == "base"


# --------------------------------------------------------------------------
# Weights + DQ threshold come from EvalConfig, not the old hardcoded literals.
# --------------------------------------------------------------------------

def test_weights_and_threshold_come_from_evalconfig(monkeypatch, tmp_path):
    captured = {}
    real_run_eval = cli_eval.run_eval

    def spy_run_eval(checkpoints, held_out, reference, embedder, transcriber, utmos,
                     weights, report_path="reports/eval.md", wer_dq_threshold=None,
                     registry=None, ref_text=""):
        captured["weights"] = weights
        captured["wer_dq_threshold"] = wer_dq_threshold
        captured["registry"] = registry
        captured["ref_text"] = ref_text
        captured["keys"] = sorted(checkpoints)
        return real_run_eval(checkpoints, held_out, reference, embedder, transcriber,
                             utmos, weights, report_path, wer_dq_threshold, registry,
                             ref_text)

    monkeypatch.setattr(cli_eval, "run_eval", spy_run_eval)
    _run(monkeypatch, tmp_path, "xtts", candidates=["xtts:finetuned=runs/xtts"])

    # EvalConfig defaults: 0.30 / 0.50 / 0.20 (NOT the old hardcoded 0.5 / 0.3 / 0.2).
    assert captured["weights"] == {"similarity": 0.30, "naturalness": 0.50, "wer": 0.20}
    assert captured["wer_dq_threshold"] == 0.20
    # A registry was threaded through so candidates get persisted.
    assert captured["registry"] is not None


# --------------------------------------------------------------------------
# UTMOS scorer is the working SpeechMOS torch.hub loader, NOT the uninstalled
# utmosv2 load_utmos (which crashes mid-run when scoring).
# --------------------------------------------------------------------------

def test_eval_uses_speechmos_utmos_loader(monkeypatch, tmp_path):
    utmos_calls: list = []
    fakes = _patch_engines(monkeypatch, "xtts")
    # _patch_loaders patches load_speechmos_utmos -- which only exists on the eval module
    # if eval imported it (the swap). The uninstalled-utmosv2 load_utmos must NOT be in
    # eval's namespace at all; if a regression reintroduces it, fail loudly.
    _patch_loaders(monkeypatch, utmos_calls=utmos_calls)
    assert not hasattr(cli_eval, "load_utmos"), \
        "eval must use load_speechmos_utmos, not the uninstalled-utmosv2 load_utmos"
    held = _heldout(tmp_path)
    cli_eval.main(["--candidates", "xtts:finetuned=runs/xtts",
                   "--reference", str(tmp_path / "ref.wav"),
                   "--held-out", held,
                   "--report", str(tmp_path / "report.md"),
                   "--register", str(tmp_path / "runs"),
                   "--device", "cpu"])
    # The SpeechMOS loader was the one invoked to obtain the score callable.
    assert len(utmos_calls) == 1
    assert len(fakes["xtts"].instances) == 1


# --------------------------------------------------------------------------
# Unknown engine token is rejected (not silently treated as gptsovits).
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# run_eval threads ref_text into every synth call as params['prompt_text'] so
# F5 (which conditions on the reference clip's transcript) gets it. Engines that
# ignore prompt_text are unaffected; default '' preserves prior empty-params behavior.
# --------------------------------------------------------------------------

class _ParamRecordingSynth:
    """F5-like fake recording the params it was called with on every synthesize."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.params_seen: list[dict] = []

    def synthesize(self, text, reference_clip, params):
        self.params_seen.append(dict(params))
        return np.full(16000, 0.5, dtype=np.float32), 16000


def test_run_eval_threads_ref_text_as_prompt_text(tmp_path):
    held = [("hello world", str(tmp_path / "r1.wav")),
            ("good morning", str(tmp_path / "r2.wav"))]
    for _, p in held:
        sf.write(p, np.random.default_rng(0).normal(0, 0.1, 22050).astype(np.float32), 22050)
    synth = _ParamRecordingSynth("runs/f5")
    run_eval({"f5_finetuned": synth}, held, str(tmp_path / "ref.wav"),
             _embedder, _transcriber, _utmos,
             {"similarity": 0.3, "naturalness": 0.5, "wer": 0.2},
             report_path=str(tmp_path / "report.md"),
             ref_text="this is the reference transcript")
    # Every held-out synth call received the reference transcript as params['prompt_text'].
    assert len(synth.params_seen) == len(held)
    assert all(p["prompt_text"] == "this is the reference transcript"
               for p in synth.params_seen)


def test_run_eval_default_ref_text_is_empty_string(tmp_path):
    held = [("hello world", str(tmp_path / "r1.wav"))]
    sf.write(held[0][1], np.random.default_rng(0).normal(0, 0.1, 22050).astype(np.float32), 22050)
    synth = _ParamRecordingSynth("runs/f5")
    run_eval({"f5_finetuned": synth}, held, str(tmp_path / "ref.wav"),
             _embedder, _transcriber, _utmos,
             {"similarity": 0.3, "naturalness": 0.5, "wer": 0.2},
             report_path=str(tmp_path / "report.md"))
    # Default preserves prior behavior: prompt_text present but empty.
    assert synth.params_seen[0]["prompt_text"] == ""


def test_unknown_engine_token_rejected(monkeypatch, tmp_path):
    _patch_engines(monkeypatch, "xtts")
    _patch_loaders(monkeypatch)
    held = _heldout(tmp_path)
    with pytest.raises((SystemExit, ValueError, KeyError)):
        cli_eval.main(["--candidates", "bogus:finetuned=runs/x",
                       "--reference", str(tmp_path / "ref.wav"),
                       "--held-out", held,
                       "--report", str(tmp_path / "r.md"),
                       "--register", str(tmp_path / "runs"),
                       "--device", "cpu"])
