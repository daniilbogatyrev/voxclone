"""GPU-free dispatch tests for scripts/serve.py multi-engine provider wiring.

`scripts.serve.make_provider(reg)` returns `provider(model)`: each call resolves
`ModelRegistry.best_checkpoint(model)` and constructs the matching *Synth via
``SYNTH_ENGINES[model](checkpoint=ckpt)`` -- it must NOT hardcode ``GPTSoVITSSynth`` for
every engine (the latent bug this task fixes). The per-engine classes are monkeypatched
with recording fakes (mirrors tests/scripts/test_cli_eval.py), so no torch / TTS /
f5_tts / chatterbox / server / GPU is touched and ``uvicorn.run`` is never called. One
test also pins the real `gptsovits` entry constructs cleanly via ``checkpoint=``.

It also pins that ``serve.app`` stays engine-agnostic: ``create_app(provider, ref)``
routes ``req.model`` through ``provider`` to ``synth.synthesize`` for any model name,
and ``SynthRequest.model`` defaults to a valid SHORT engine key.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import scripts.serve as serve
from voxclone.common.registry import ModelRegistry
from voxclone.serve.app import SynthRequest, create_app
from voxclone.synth import SYNTH_ENGINES, GPTSoVITSSynth


# --------------------------------------------------------------------------
# A recording fake mirroring the checkpoint-bound SynthAdapter contract
# (ctor takes checkpoint=; synthesize(text, ref, params) -> (audio, sr)).
# This is the same GPU-free pattern as tests/scripts/test_cli_eval.py.
# --------------------------------------------------------------------------
class FakeSynth:
    instances: list["FakeSynth"] = []

    def __init__(self, checkpoint, **kwargs):
        self.checkpoint = checkpoint
        self.kwargs = kwargs
        type(self).instances.append(self)

    def synthesize(self, text, reference_clip, params):
        return np.zeros(2400, dtype=np.float32), 24000


def _patch_engines(monkeypatch, *engines):
    """Replace each named engine in SYNTH_ENGINES with a distinct recording fake."""
    FakeSynth.instances = []
    fakes = {}
    for eng in engines:
        cls = type(f"Fake_{eng}", (FakeSynth,), {"instances": []})
        fakes[eng] = cls
        monkeypatch.setitem(serve.SYNTH_ENGINES, eng, cls)
    return fakes


def _registry_with(tmp_path, **model_to_ckpt):
    """A real ModelRegistry pre-populated via register() (round-trip on disk).

    Keys are the COMPOUND ``<engine>_<label>`` keys eval really writes (e.g.
    ``xtts_finetuned``), so ``score=1.0`` by default keeps a single winner per engine.
    """
    reg = ModelRegistry(str(tmp_path))
    for model, ckpt in model_to_ckpt.items():
        reg.register(model, ckpt, {"score": 1.0})
    return reg


def _registry_scored(tmp_path, *entries):
    """Like ``_registry_with`` but each entry is ``(key, ckpt, score)`` so a single engine
    can have several compound-key candidates with distinct scores (winner = max score)."""
    reg = ModelRegistry(str(tmp_path))
    for key, ckpt, score in entries:
        reg.register(key, ckpt, {"score": score})
    return reg


# --------------------------------------------------------------------------
# provider(model) builds the RIGHT *Synth per model via SYNTH_ENGINES + registry
# (was always GPTSoVITSSynth before this task).
# --------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["xtts", "f5", "gptsovits", "chatterbox"])
def test_provider_builds_correct_synth_per_model(monkeypatch, tmp_path, model):
    fakes = _patch_engines(monkeypatch, model)
    ckpt = str(tmp_path / f"runs/{model}/best")
    # eval registers under the COMPOUND key <engine>_<label>; serve resolves by SHORT key.
    reg = _registry_with(tmp_path, **{f"{model}_finetuned": ckpt})
    synth = serve.make_provider(reg)(model)
    # Built from SYNTH_ENGINES[model] -- the per-engine class, NOT always GPTSoVITSSynth.
    assert isinstance(synth, fakes[model])
    assert synth.checkpoint == ckpt


def test_provider_selects_only_the_requested_engine(monkeypatch, tmp_path):
    fakes = _patch_engines(monkeypatch, "xtts", "f5", "gptsovits", "chatterbox")
    reg = _registry_with(tmp_path, xtts_finetuned=str(tmp_path / "runs/xtts/best"))
    serve.make_provider(reg)("xtts")
    assert len(fakes["xtts"].instances) == 1
    # No other engine's class was constructed (no hardcoded GPTSoVITSSynth fallback).
    for other in ("f5", "gptsovits", "chatterbox"):
        assert fakes[other].instances == []


def test_provider_binds_best_checkpoint_from_registry(monkeypatch, tmp_path):
    _patch_engines(monkeypatch, "f5")
    ckpt = str(tmp_path / "runs/f5/ckpt_step_400")
    reg = _registry_with(tmp_path, f5_finetuned=ckpt)
    synth = serve.make_provider(reg)("f5")
    # The synth is bound to the compound-key winner serve resolves for the engine.
    assert synth.checkpoint == ckpt


def test_provider_resolves_compound_key_winner_for_short_engine(monkeypatch, tmp_path):
    # eval writes compound keys (engine_label); serve gets the SHORT engine key. The provider
    # must pick the highest-score entry among that engine's compound candidates.
    _patch_engines(monkeypatch, "gptsovits")
    ft = str(tmp_path / "runs/gptsovits/best")
    reg = _registry_scored(
        tmp_path,
        ("gptsovits_finetuned", ft, 0.91),
        ("gptsovits_zeroshot", "v2Pro", 0.40),
    )
    synth = serve.make_provider(reg)("gptsovits")
    # The finetuned candidate wins on score, even though zeroshot was registered too.
    assert synth.checkpoint == ft


def test_provider_winner_ignores_other_engines_compound_keys(monkeypatch, tmp_path):
    # An engine prefix match must be on the WHOLE short key, not a substring: requesting
    # "f5" must not match "f5x_*"-style keys, and other engines' winners are irrelevant.
    fakes = _patch_engines(monkeypatch, "f5", "xtts")
    f5_ckpt = str(tmp_path / "runs/f5/best")
    reg = _registry_scored(
        tmp_path,
        ("xtts_finetuned", str(tmp_path / "runs/xtts/best"), 0.99),  # higher score, wrong engine
        ("f5_finetuned", f5_ckpt, 0.50),
    )
    synth = serve.make_provider(reg)("f5")
    assert isinstance(synth, fakes["f5"])
    assert synth.checkpoint == f5_ckpt
    assert fakes["xtts"].instances == []


def test_provider_unknown_model_raises_keyerror(monkeypatch, tmp_path):
    _patch_engines(monkeypatch, "xtts")
    reg = _registry_with(tmp_path, xtts_finetuned=str(tmp_path / "runs/xtts/best"))
    provider = serve.make_provider(reg)
    with pytest.raises(KeyError):
        provider("f5")          # nothing registered for f5 -> no winner -> KeyError


def test_provider_empty_registry_raises_keyerror(monkeypatch, tmp_path):
    _patch_engines(monkeypatch, "xtts")
    reg = ModelRegistry(str(tmp_path))   # empty: no candidates at all
    provider = serve.make_provider(reg)
    with pytest.raises(KeyError):
        provider("xtts")


def test_provider_caches_synth_per_model(monkeypatch, tmp_path):
    fakes = _patch_engines(monkeypatch, "xtts")
    reg = _registry_with(tmp_path, xtts_finetuned=str(tmp_path / "runs/xtts/best"))
    provider = serve.make_provider(reg)
    assert provider("xtts") is provider("xtts")     # same instance reused
    assert len(fakes["xtts"].instances) == 1        # constructed exactly once


def test_real_gptsovits_entry_constructs_via_checkpoint_kwarg(tmp_path):
    # No monkeypatch: the real SYNTH_ENGINES["gptsovits"] must accept checkpoint= (GPU-free
    # ctor; the heavy server call only happens inside synthesize, never here).
    reg = _registry_with(tmp_path, gptsovits_finetuned=str(tmp_path / "runs/gptsovits/best"))
    synth = serve.make_provider(reg)("gptsovits")
    assert isinstance(synth, GPTSoVITSSynth)
    assert synth.checkpoint == str(tmp_path / "runs/gptsovits/best")


# --------------------------------------------------------------------------
# Round trip: create_app(provider, ref) routes req.model -> provider -> synthesize.
# app.py stays engine-agnostic (no engine names baked into create_app).
# --------------------------------------------------------------------------

def test_create_app_routes_request_model_through_provider():
    seen = {}

    class RecordingSynth:
        def __init__(self, model):
            self.model = model

        def synthesize(self, text, reference_clip, params):
            seen.update(model=self.model, text=text, ref=reference_clip)
            return np.zeros(2400, dtype=np.float32), 24000

    app = create_app(lambda model: RecordingSynth(model), reference_clip="ref.wav")
    resp = TestClient(app).post("/synthesize", json={"text": "hi", "model": "xtts"})
    assert resp.status_code == 200
    # the engine-agnostic app passed req.model straight to provider, then synthesize ran.
    assert seen == {"model": "xtts", "text": "hi", "ref": "ref.wav"}


def test_create_app_threads_ref_text_as_prompt_text(monkeypatch):
    # F5 needs the reference clip's transcript as params['prompt_text']. create_app's
    # configured ref_text must reach synthesize via params (engines that ignore it are
    # unaffected), unless the request overrides it.
    seen = {}

    class F5LikeSynth:
        def synthesize(self, text, reference_clip, params):
            seen["params"] = dict(params)
            return np.zeros(2400, dtype=np.float32), 24000

    app = create_app(lambda model: F5LikeSynth(), reference_clip="ref.wav",
                     ref_text="the held-out reference transcript")
    resp = TestClient(app).post("/synthesize", json={"text": "hi", "model": "f5"})
    assert resp.status_code == 200
    assert seen["params"]["prompt_text"] == "the held-out reference transcript"


def test_create_app_request_ref_text_overrides_app_default(monkeypatch):
    seen = {}

    class F5LikeSynth:
        def synthesize(self, text, reference_clip, params):
            seen["params"] = dict(params)
            return np.zeros(2400, dtype=np.float32), 24000

    app = create_app(lambda model: F5LikeSynth(), reference_clip="ref.wav",
                     ref_text="app-level default transcript")
    resp = TestClient(app).post(
        "/synthesize",
        json={"text": "hi", "model": "f5", "ref_text": "per-request transcript"})
    assert resp.status_code == 200
    assert seen["params"]["prompt_text"] == "per-request transcript"


def test_create_app_default_ref_text_is_empty_string(monkeypatch):
    # Default preserves prior behavior: prompt_text is the empty string when no ref_text set.
    seen = {}

    class F5LikeSynth:
        def synthesize(self, text, reference_clip, params):
            seen["params"] = dict(params)
            return np.zeros(2400, dtype=np.float32), 24000

    app = create_app(lambda model: F5LikeSynth(), reference_clip="ref.wav")
    resp = TestClient(app).post("/synthesize", json={"text": "hi", "model": "f5"})
    assert resp.status_code == 200
    assert seen["params"]["prompt_text"] == ""


def test_end_to_end_make_provider_through_create_app(monkeypatch, tmp_path):
    # Wire the real make_provider (with a monkeypatched engine class) through create_app.
    _patch_engines(monkeypatch, "xtts")
    ckpt = str(tmp_path / "runs/xtts/best")
    reg = _registry_with(tmp_path, xtts_finetuned=ckpt)
    provider = serve.make_provider(reg)

    app = create_app(provider, reference_clip="ref.wav")
    resp = TestClient(app).post("/synthesize", json={"text": "hello world", "model": "xtts"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    # the synth the app used is the checkpoint-bound one make_provider built.
    assert provider("xtts").checkpoint == ckpt


def test_create_app_unknown_model_is_404(monkeypatch, tmp_path):
    _patch_engines(monkeypatch, "xtts")
    reg = _registry_with(tmp_path, xtts_finetuned=str(tmp_path / "runs/xtts/best"))
    app = create_app(serve.make_provider(reg), reference_clip="ref.wav")
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/synthesize", json={"text": "hi", "model": "f5"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# app.py change: SynthRequest default model is a valid SHORT engine key only.
# --------------------------------------------------------------------------

def test_synth_request_default_model_is_a_valid_short_engine_key():
    req = SynthRequest(text="hi")
    assert req.model in SYNTH_ENGINES
    assert req.model in {"xtts", "f5", "gptsovits", "chatterbox"}
