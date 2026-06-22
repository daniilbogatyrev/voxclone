"""GPU-free end-to-end orchestration smoke: prep->train->eval->register->serve.

This test pins the load-bearing *naming contract* across the whole bake-off chain by
running a mocked round-trip through the THREE real CLI modules
(``scripts.train`` / ``scripts.eval`` / ``scripts.serve``) plus the real
``voxclone.eval.runner.run_eval`` + ``voxclone.common.registry.ModelRegistry`` +
``voxclone.synth.SYNTH_ENGINES``. Nothing heavy runs: the trainer's subprocess
``runner`` is a fake, each synth's ``generate_fn`` is a fake (no server / GPU), and the
eval model loaders are trivial stand-ins. No torch / TTS / f5_tts / chatterbox / whisper
is imported, no real server is launched, no model weights are loaded, no network call.

The single SHORT engine key (e.g. ``"xtts"``) is simultaneously, end-to-end:

    --engine value  ==  --out basename  ==  train checkpoint dir basename
                    ==  SYNTH_ENGINES key  ==  eval candidate engine token
                    ==  ModelRegistry model key  ==  serve provider(model) lookup key
                    ==  SynthRequest.model default-valid key.

The acceptance bar: a mocked train->eval(register)->serve round-trip resolves the same
engine key end-to-end -- no ``KeyError``, ``best_checkpoint`` is non-None, and the served
synth is the engine's own checkpoint-bound class bound to the trained checkpoint dir.

FIXED CROSS-WAVE BUG (Plan 07, ``src/voxclone/synth/__init__.py``): the SYNTH_ENGINES map
was written (Wave 2) BEFORE the checkpoint-bound ``XTTSSynth``/``F5Synth`` classes existed
(Wave 3), so it aliased the NOTEBOOK-facing ``XTTSAdapter``/``F5Adapter`` (ctor
``(model_dir/model/device, _synth)``) -- which have no ``checkpoint=`` kwarg, so the real
serve/eval contract ``SYNTH_ENGINES[engine](checkpoint=...)`` only worked for ``gptsovits``.
The per-stage CLI unit tests missed it because they monkeypatch the engine classes with
fakes -- exactly the gap an e2e smoke exists to surface. The map now imports the
checkpoint-bound ``{XTTSSynth,F5Synth,ChatterboxSynth}``; the regression is asserted
positively in ``test_synth_map_uses_checkpoint_bound_class_not_notebook_adapter`` below.
"""
import os
from pathlib import Path

import numpy as np
import pytest

import scripts.train as cli_train
import scripts.eval as cli_eval
import scripts.serve as cli_serve
from voxclone.common.registry import ModelRegistry
from voxclone.eval.runner import run_eval
from voxclone.synth import SYNTH_ENGINES
from voxclone.synth.xtts import XTTSSynth          # eval/serve-facing, checkpoint-bound
from voxclone.synth.f5 import F5Synth              # eval/serve-facing, checkpoint-bound
from voxclone.synth.gptsovits import GPTSoVITSSynth
from voxclone.train.base import TrainResult

# The eval/serve-facing checkpoint-bound synth class per engine (what serve MUST build).
# This is what SYNTH_ENGINES *should* map to (see the KNOWN UPSTREAM BUG note above).
CHECKPOINT_BOUND_SYNTH = {"xtts": XTTSSynth, "f5": F5Synth, "gptsovits": GPTSoVITSSynth}


# --------------------------------------------------------------------------
# GPU-free seams. Each mirrors the real adapter contract WITHOUT heavy deps:
#   * FakeTrainer.train writes a checkpoint dir (like XTTSTrainer.train), so the
#     downstream stages have a real on-disk artifact to thread.
#   * fake_generate replaces the synth's server/GPU call (the synth generate_fn),
#     returning a deterministic non-silent clip the eval front end can normalize.
#   * the eval model loaders (embedder/transcriber/utmos) are trivial stand-ins.
# --------------------------------------------------------------------------
class FakeTrainer:
    """Mirrors the TrainAdapter contract; writes a checkpoint dir on .train()."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train(self, manifest_path, out_dir, config) -> TrainResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "best_model.pth").write_text("fake-weights")   # the checkpoint artifact
        return TrainResult(checkpoint_dir=str(out), steps=int(config.get("epochs", 7)))


def _fake_generate(text, reference_clip, checkpoint, params):
    # Deterministic, non-silent 16 kHz mono so the eval front end (loudness+resample)
    # and similarity/utmos are well-defined. NO server, NO GPU, NO network.
    return np.full(16000, 0.5, dtype=np.float32), 16000


def _embedder(audio):
    assert isinstance(audio, np.ndarray)
    return np.array([1.0, 0.0])


def _transcriber(audio, sr):
    return [{"word": "hello", "probability": 1.0}, {"word": "world", "probability": 1.0}]


def _utmos(audio, sr):
    return 4.5


def _cfg(tmp_path) -> str:
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  epochs: 7\n")
    return str(p)


def _heldout(tmp_path):
    """A tiny held-out list[(text, real_clip)] with two real clips on disk."""
    import soundfile as sf

    clips = []
    for i in range(2):
        p = tmp_path / f"real{i}.wav"
        sig = np.random.default_rng(i).normal(0, 0.1, 22050).astype(np.float32)
        sf.write(str(p), sig, 22050)
        clips.append(p)
    return [("hello world", str(clips[0])), ("good morning", str(clips[1]))]


# --------------------------------------------------------------------------
# Stage helpers, each driving the REAL CLI/module surface for one stage.
# --------------------------------------------------------------------------
def _train_stage(monkeypatch, tmp_path, engine: str) -> str:
    """Run scripts.train.main for ``engine`` with a fake trainer. Returns the --out dir
    (whose basename == engine key, the naming-contract anchor)."""
    monkeypatch.setitem(cli_train.TRAIN_ENGINES, engine, FakeTrainer)
    out = str(tmp_path / "runs" / engine)         # basename == engine key
    argv = ["--engine", engine, "--manifest", str(tmp_path / "m.jsonl"),
            "--config", _cfg(tmp_path), "--out", out]
    if engine == "gptsovits":
        argv += ["--gptsovits-root", "/opt/GPT-SoVITS"]   # CLI-required root (trainer is faked)
    cli_train.main(argv)
    return out


def _eval_register_stage(tmp_path, engine: str, synth) -> ModelRegistry:
    """Score+register the trained checkpoint under the SHORT engine key via the real
    voxclone.eval.runner.run_eval + ModelRegistry (the same core scripts.eval drives).
    ``synth`` is the engine's checkpoint-bound adapter (heavy server call stubbed).

    NOTE: this stage keys the candidate by the BARE engine key. That is a deliberately
    simplified anchor for the SynthAdapter/SYNTH_ENGINES naming-contract checks; it is NOT
    how ``scripts.eval`` actually registers (it uses COMPOUND ``<engine>_<label>`` keys).
    For the *real* eval->serve compound-key roundtrip use ``_real_eval_register_stage``.
    """
    reg = ModelRegistry(str(tmp_path / "runs"))
    # checkpoints keyed by the SHORT engine key == registry model key == serve lookup key.
    run_eval({engine: synth}, _heldout(tmp_path), str(tmp_path / "ref.wav"),
             _embedder, _transcriber, _utmos,
             {"similarity": 0.30, "naturalness": 0.50, "wer": 0.20},
             report_path=str(tmp_path / "report.md"),
             wer_dq_threshold=0.20, registry=reg)
    return reg


def _real_eval_register_stage(monkeypatch, tmp_path, engine: str, ckpt: str,
                              label: str = "finetuned") -> ModelRegistry:
    """Drive the REAL ``scripts.eval`` registration path end-to-end (no bypass).

    This mirrors exactly what ``scripts.eval.main``'s candidate loop does -- it runs the
    REAL ``cli_eval._parse_candidate`` to split the ``engine:label=checkpoint`` token into
    the COMPOUND registry key ``<engine>_<label>`` + bare engine + checkpoint, builds the
    engine's adapter from the LIVE ``SYNTH_ENGINES`` map bound to that checkpoint (server
    call stubbed via ``generate_fn``), and registers it through the REAL
    ``voxclone.eval.runner.run_eval`` + ``ModelRegistry``. So the candidate lands under the
    COMPOUND key (e.g. ``xtts_finetuned``) WITH a ``score`` -- precisely what serve's
    ``best_for_engine(engine)`` must resolve. The heavy ``SYNTH_ENGINES[engine](checkpoint=)``
    default ctor would bind ``_real_generate`` (a server POST); we monkeypatch the adapter's
    ``generate_fn`` so no server/GPU is touched.
    """
    name, parsed_engine, parsed_ckpt = cli_eval._parse_candidate(f"{engine}:{label}={ckpt}")
    assert parsed_engine == engine and parsed_ckpt == ckpt        # real token parse, not a stub
    assert name == f"{engine}_{label}"                            # COMPOUND registry/report key

    synth = SYNTH_ENGINES[engine](checkpoint=parsed_ckpt)         # live map, real ctor
    monkeypatch.setattr(synth, "generate_fn", _fake_generate)     # stub the server/GPU call only

    reg = ModelRegistry(str(tmp_path / "runs"))
    # The candidate is keyed by the COMPOUND name (engine_label) -- exactly as scripts.eval
    # registers it; serve resolves the bare engine key to this via best_for_engine.
    run_eval({name: synth}, _heldout(tmp_path), str(tmp_path / "ref.wav"),
             _embedder, _transcriber, _utmos,
             {"similarity": 0.30, "naturalness": 0.50, "wer": 0.20},
             report_path=str(tmp_path / "report.md"),
             wer_dq_threshold=0.20, registry=reg)
    return reg


# --------------------------------------------------------------------------
# The end-to-end naming-contract round-trip (worked example: gptsovits, whose
# real SYNTH_ENGINES entry constructs cleanly via checkpoint=, so the WHOLE
# chain -- including the real scripts.serve.make_provider over the live map --
# resolves the one engine key end-to-end).
# --------------------------------------------------------------------------
def test_e2e_naming_contract_roundtrip_through_live_synth_map():
    """train -> eval(register) -> serve resolves the SAME engine key end-to-end via the
    REAL SYNTH_ENGINES map + real scripts.serve.make_provider (no KeyError; ckpt non-None).
    Uses gptsovits, the engine whose live map entry is checkpoint-bound today."""
    import tempfile

    engine = "gptsovits"
    monkeypatch = pytest.MonkeyPatch()
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            # 1) TRAIN: writes a checkpoint dir whose basename == the engine key.
            ckpt = _train_stage(monkeypatch, tmp, engine)
            assert Path(ckpt).is_dir() and Path(ckpt).name == engine     # --out basename == key
            assert (Path(ckpt) / "best_model.pth").exists()              # a real ckpt artifact

            # 2) EVAL + REGISTER: built from the LIVE SYNTH_ENGINES map (the thing under
            #    test), checkpoint-bound to the trained dir, server call stubbed.
            synth = SYNTH_ENGINES[engine](checkpoint=ckpt, generate_fn=_fake_generate)
            reg = _eval_register_stage(tmp, engine, synth)
            assert engine in reg._load()                                # registered under key
            best = reg.best_checkpoint(engine)
            assert best is not None and best == ckpt                    # ACCEPTANCE: non-None

            # 3) SERVE: the REAL provider resolves best_checkpoint and builds the engine's
            #    own checkpoint-bound class -- no KeyError, same key resolved end-to-end.
            served = cli_serve.make_provider(reg)(engine)               # ACCEPTANCE: no KeyError
            assert isinstance(served, GPTSoVITSSynth)                   # engine's own synth class
            assert served.checkpoint == ckpt == best                   # one key, end-to-end
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize("engine,bound_cls", [
    ("xtts", XTTSSynth), ("f5", F5Synth), ("gptsovits", GPTSoVITSSynth)])
def test_e2e_real_eval_compound_key_registration_resolves_through_serve(
        monkeypatch, tmp_path, engine, bound_cls):
    """train -> REAL eval (COMPOUND-key registration) -> REAL serve resolution, no bypass.

    The previous live-map roundtrip registered the candidate under the BARE engine key, so
    serve resolved it trivially (key == engine) and never exercised the real eval->serve
    contract: eval registers under the COMPOUND ``<engine>_<label>`` key (``xtts_finetuned``),
    and serve resolves the BARE engine key to it via ``best_for_engine``. This case drives
    the REAL ``cli_eval._parse_candidate`` + ``run_eval`` registration (compound key, WITH a
    score) and the REAL ``cli_serve.make_provider(reg)(engine)`` resolution, asserting the
    served winner is the engine's own checkpoint-bound class bound to the trained dir.
    """
    # 1) TRAIN: checkpoint dir, basename == engine key (the naming-contract anchor).
    ckpt = _train_stage(monkeypatch, tmp_path, engine)
    assert Path(ckpt).is_dir() and Path(ckpt).name == engine

    # 2) REAL EVAL: parse the engine:label=checkpoint token, build from the LIVE map, and
    #    register the candidate under the COMPOUND key (engine_finetuned) WITH a score.
    reg = _real_eval_register_stage(monkeypatch, tmp_path, engine, ckpt, label="finetuned")
    data = reg._load()
    assert f"{engine}_finetuned" in data                              # COMPOUND key (real path)
    assert engine not in data                                          # NOT the bare-key bypass
    assert "score" in data[f"{engine}_finetuned"][0]["metrics"]       # scored, for best_for_engine

    # 3) REAL SERVE: resolve the BARE engine key -> the compound winner's checkpoint via
    #    best_for_engine, building the engine's own checkpoint-bound class. No KeyError.
    served = cli_serve.make_provider(reg)(engine)                     # ACCEPTANCE: no KeyError
    assert isinstance(served, bound_cls)                             # engine's own synth class
    assert served.checkpoint == ckpt                                 # the trained winner ckpt
    # cross-check serve resolves the SAME (compound key, checkpoint) the registry holds.
    assert reg.best_for_engine(engine) == (f"{engine}_finetuned", ckpt)


def test_e2e_real_eval_serve_picks_highest_score_label_for_engine(monkeypatch, tmp_path):
    """When an engine has multiple eval candidates (compound labels), REAL serve resolves the
    BARE engine key to the HIGHEST-score one -- proving best_for_engine, not a bare-key alias,
    drives resolution. Two xtts candidates (finetuned vs zeroshot) score differently because
    they bind different checkpoints; serve must return the winning label's checkpoint."""
    engine = "xtts"
    ft_ckpt = _train_stage(monkeypatch, tmp_path, engine)            # runs/xtts (the trained dir)
    zs_ckpt = str(tmp_path / "runs" / "xtts_base")                   # a distinct zero-shot ckpt
    Path(zs_ckpt).mkdir(parents=True, exist_ok=True)

    reg = ModelRegistry(str(tmp_path / "runs"))
    # Register both compound candidates through the registry with explicit, distinct scores
    # (run_eval would derive these; here we pin them so the winner is unambiguous).
    reg.register("xtts_zeroshot", zs_ckpt, {"score": 0.10})
    reg.register("xtts_finetuned", ft_ckpt, {"score": 0.90})         # the higher-score winner

    served = cli_serve.make_provider(reg)(engine)                    # resolve BARE key
    assert isinstance(served, XTTSSynth)
    assert served.checkpoint == ft_ckpt                             # the winning label's ckpt
    assert reg.best_for_engine(engine) == ("xtts_finetuned", ft_ckpt)


@pytest.mark.parametrize("engine", ["xtts", "f5", "gptsovits"])
def test_e2e_roundtrip_holds_for_every_trainable_engine(monkeypatch, tmp_path, engine):
    """The naming-contract round-trip holds identically for each fine-tunable engine
    using its eval/serve-facing checkpoint-bound synth class (XTTSSynth/F5Synth/
    GPTSoVITSSynth -- what serve MUST build). The registry key == --out basename ==
    engine key threads train->register->resolve with no KeyError, best_checkpoint non-None.
    """
    # 1) TRAIN -> checkpoint dir, basename == engine key.
    ckpt = _train_stage(monkeypatch, tmp_path, engine)
    assert Path(ckpt).name == engine

    # 2) EVAL + REGISTER under the engine key, using the checkpoint-bound synth class.
    synth = CHECKPOINT_BOUND_SYNTH[engine](checkpoint=ckpt, generate_fn=_fake_generate)
    reg = _eval_register_stage(tmp_path, engine, synth)
    best = reg.best_checkpoint(engine)
    assert best == ckpt and best is not None                          # ACCEPTANCE: non-None

    # 3) RESOLVE the served winner by the SAME engine key (registry round-trips on disk).
    #    A fresh registry reads back the persisted entry -> proves the key threads on disk.
    reg2 = ModelRegistry(str(tmp_path / "runs"))
    resolved_ckpt = reg2.best_checkpoint(engine)                      # ACCEPTANCE: no KeyError path
    assert resolved_ckpt == ckpt
    rebuilt = CHECKPOINT_BOUND_SYNTH[engine](checkpoint=resolved_ckpt, generate_fn=_fake_generate)
    assert rebuilt.checkpoint == ckpt                                 # one key, end-to-end


@pytest.mark.parametrize("engine,bound_cls", [("xtts", XTTSSynth), ("f5", F5Synth)])
def test_synth_map_uses_checkpoint_bound_class_not_notebook_adapter(engine, bound_cls):
    """Regression (was a PINNED cross-wave bug, now FIXED): SYNTH_ENGINES[engine] must be the
    checkpoint-bound eval/serve class so the live serve/eval contract
    ``SYNTH_ENGINES[engine](checkpoint=...)`` constructs -- NOT the notebook-facing ``*Adapter``
    (whose ctor has no ``checkpoint=`` kwarg). Fixed by importing the checkpoint-bound
    ``{XTTSSynth,F5Synth}`` (not the ``*Adapter`` aliases) into ``synth/__init__.py``."""
    assert SYNTH_ENGINES[engine] is bound_cls, (
        f"SYNTH_ENGINES[{engine!r}] is {SYNTH_ENGINES[engine].__name__}, "
        f"expected checkpoint-bound {bound_cls.__name__}")
    synth = SYNTH_ENGINES[engine](
        checkpoint="runs/" + engine,
        generate_fn=lambda text, ref, ckpt, params: (np.zeros(8, dtype=np.float32), 24000),
    )
    assert synth.checkpoint == "runs/" + engine               # checkpoint bound at construction
    audio, sr = synth.synthesize("hi", "ref.wav", {})         # empty-params eval path works
    assert sr == 24000 and audio.dtype == np.float32


def test_serve_resolves_finetuned_winner_via_real_make_provider(tmp_path):
    """The real scripts.serve.make_provider over a registered gptsovits winner resolves the
    BARE engine key to a checkpoint-bound GPTSoVITSSynth -- the serve half of the contract, no
    monkeypatch (the latent 'always GPTSoVITSSynth' / wrong-key bugs would surface here).

    The winner is registered under the COMPOUND ``gptsovits_finetuned`` key exactly as
    ``scripts.eval`` does; serve must resolve the bare ``gptsovits`` key to it via
    ``best_for_engine`` (registering under the bare key would be the OLD bypass)."""
    ckpt = str(tmp_path / "runs" / "gptsovits")
    reg = ModelRegistry(str(tmp_path / "runs"))
    reg.register("gptsovits_finetuned", ckpt, {"score": 0.9})         # COMPOUND key, as eval writes
    assert reg.best_checkpoint("gptsovits") is None                  # bare-key bypass would NOT find it
    synth = cli_serve.make_provider(reg)("gptsovits")                # serve resolves via best_for_engine
    assert isinstance(synth, GPTSoVITSSynth)
    assert synth.checkpoint == ckpt


def test_train_out_basename_equals_train_map_key(monkeypatch, tmp_path):
    """The --out basename the trainer received is exactly the TRAIN_ENGINES / engine key."""
    engine = "xtts"
    assert engine in cli_train.TRAIN_ENGINES                          # the trainable key
    captured = {}

    class _Recorder(FakeTrainer):
        def train(self, manifest_path, out_dir, config):
            captured["out_dir"] = out_dir
            return super().train(manifest_path, out_dir, config)

    monkeypatch.setitem(cli_train.TRAIN_ENGINES, engine, _Recorder)
    out = str(tmp_path / "runs" / engine)
    cli_train.main(["--engine", engine, "--manifest", str(tmp_path / "m.jsonl"),
                    "--config", _cfg(tmp_path), "--out", out])
    assert captured["out_dir"] == out
    assert Path(captured["out_dir"]).name == engine                  # basename == engine key


def test_eval_candidate_engine_token_matches_short_synth_key():
    """scripts.eval parses the candidate token's engine half to the SHORT SYNTH_ENGINES
    key -- the SAME key train writes and serve resolves (the candidate-token half of the
    naming contract)."""
    name, engine, checkpoint = cli_eval._parse_candidate("xtts:finetuned=runs/xtts")
    assert engine == "xtts" and engine in SYNTH_ENGINES              # candidate engine == key
    assert checkpoint == "runs/xtts"                                 # =checkpoint half binds
    assert name == "xtts_finetuned"                                  # registry/report key


def test_serve_unknown_key_raises_keyerror(tmp_path):
    """A key with nothing registered does NOT resolve -- the contract's negative case
    (proves the positive resolution above is real, not a silent fallback)."""
    reg = ModelRegistry(str(tmp_path / "runs"))                      # empty registry
    provider = cli_serve.make_provider(reg)
    with pytest.raises(KeyError):
        provider("gptsovits")                                       # best_checkpoint None -> KeyError


# The heavy MODEL libs that the GPU-free e2e must NEVER load (no GPU, no weights, no net).
# uvicorn is intentionally excluded: scripts.serve imports it at module top, but the real
# uvicorn.run launch is pragma:no-cover and is never executed here (no server is started).
HEAVY = ("torch", "TTS", "f5_tts", "chatterbox", "whisper", "speechbrain")

# The full REAL train->eval(compound register)->serve chain, driven entirely from this
# module's GPU-free seams. Run in a CLEAN subprocess (below) so the heavy-import assertion
# is ABSOLUTE ABSENCE, not an order-dependent before/after diff of the shared interpreter.
_GPU_FREE_DRIVER = '''
import sys, tempfile
from pathlib import Path

import scripts.train as cli_train
import scripts.eval as cli_eval
import scripts.serve as cli_serve
from voxclone.common.registry import ModelRegistry
from voxclone.eval.runner import run_eval
from voxclone.synth import SYNTH_ENGINES

# Reuse THIS test module's GPU-free fakes so the subprocess drives the same real chain.
import tests.scripts.test_e2e_orchestration as T

HEAVY = {0!r}
engine = "gptsovits"
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    # TRAIN (real CLI, fake trainer) -> checkpoint dir.
    cli_train.TRAIN_ENGINES[engine] = T.FakeTrainer
    (tmp / "c.yaml").write_text("train:\\n  epochs: 7\\n")
    out = str(tmp / "runs" / engine)
    cli_train.main(["--engine", engine, "--manifest", str(tmp / "m.jsonl"),
                    "--config", str(tmp / "c.yaml"), "--out", out,
                    "--gptsovits-root", "/opt/GPT-SoVITS"])
    # REAL eval: parse the compound candidate token, build from the live map, register.
    name, eng, ckpt = cli_eval._parse_candidate(engine + ":finetuned=" + out)
    synth = SYNTH_ENGINES[eng](checkpoint=ckpt)
    synth.generate_fn = T._fake_generate          # stub the server/GPU call only
    reg = ModelRegistry(str(tmp / "runs"))
    run_eval({{name: synth}}, T._heldout(tmp), str(tmp / "ref.wav"),
             T._embedder, T._transcriber, T._utmos,
             {{"similarity": 0.30, "naturalness": 0.50, "wer": 0.20}},
             report_path=str(tmp / "report.md"), wer_dq_threshold=0.20, registry=reg)
    # REAL serve resolution over the compound winner.
    served = cli_serve.make_provider(reg)(engine)
    assert served.checkpoint == out

present = sorted(h for h in HEAVY if h in sys.modules)
# In a CLEAN interpreter, the GPU-free chain must leave EVERY heavy ML lib ABSENT.
assert not present, "GPU-free e2e imported heavy ML libs: " + repr(present)
print("GPU_FREE_OK")
'''


def test_e2e_is_gpu_free_no_heavy_imports():
    """Sanity guard: the whole real train->eval->serve round-trip runs WITHOUT importing any
    heavy ML lib (no torch/TTS/f5_tts/chatterbox/whisper/speechbrain), loading weights, or
    hitting a GPU/network.

    Driven in a CLEAN subprocess so the assertion is ABSOLUTE ABSENCE and order-independent:
    the prior before/after ``sys.modules`` diff was order-dependent because an UNRELATED
    earlier test (e.g. ``set_seed`` legitimately importing ``torch``) pre-pollutes the shared
    interpreter, forcing the weaker "no NEW heavy import" phrasing. A fresh interpreter starts
    with none of them loaded, so we can assert NONE is present after the full chain -- failing
    loudly the instant any e2e stage imports a model lib / loads weights / touches a GPU.
    """
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    # Match pytest's pythonpath = ["src", "."]: src for scripts/voxclone, root for tests.*.
    extra = [str(repo_root / "src"), str(repo_root)]
    env["PYTHONPATH"] = os.pathsep.join(extra + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    proc = subprocess.run(
        [sys.executable, "-c", _GPU_FREE_DRIVER.format(HEAVY)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        "GPU-free e2e subprocess failed:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "GPU_FREE_OK" in proc.stdout, f"driver did not complete:\n{proc.stdout}\n{proc.stderr}"
