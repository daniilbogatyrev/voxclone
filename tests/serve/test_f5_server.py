import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient


class _FakeF5Model:
    """GPU-free stand-in for the real f5_tts.api.F5TTS model.

    Records the checkpoint binding from /load_checkpoint and the tts call args so the
    test can assert the server wires the reference clip + prompt_text transcript through.
    """

    sr = 24000

    def __init__(self):
        self.loaded = None
        self.tts_calls = []

    def load_checkpoint(self, checkpoint_dir, vocab_file=None):
        self.loaded = (checkpoint_dir, vocab_file)

    def tts(self, text, reference_clip, prompt_text="", **kw):
        self.tts_calls.append({"text": text, "reference_clip": reference_clip,
                               "prompt_text": prompt_text, **kw})
        return np.zeros(24000, dtype=np.float32), self.sr


def _client(fake):
    from voxclone.serve.f5_server import make_app
    return TestClient(make_app(model_factory=lambda: fake))


def test_health_endpoint():
    # Identity-aware readiness: the studio polls GET /health before driving the server.
    r = _client(_FakeF5Model()).get("/health")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_load_checkpoint_binds_finetuned_safetensors():
    fake = _FakeF5Model()
    c = _client(fake)
    r = c.post("/load_checkpoint",
               json={"checkpoint_dir": "runs/f5", "vocab_file": "runs/f5/vocab.txt"})
    assert r.status_code == 200
    assert r.json()["loaded"] == "runs/f5"
    assert fake.loaded == ("runs/f5", "runs/f5/vocab.txt")


def test_load_checkpoint_vocab_file_optional():
    fake = _FakeF5Model()
    c = _client(fake)
    r = c.post("/load_checkpoint", json={"checkpoint_dir": "runs/f5"})
    assert r.status_code == 200
    assert fake.loaded == ("runs/f5", None)


def test_tts_round_trips_24k_float32_mono_wav():
    fake = _FakeF5Model()
    c = _client(fake)
    assert c.post("/load_checkpoint", json={"checkpoint_dir": "runs/f5"}).status_code == 200
    r = c.post("/tts", json={"text": "hello there", "reference_clip": "ref.wav",
                             "prompt_text": "the enrollment transcript"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")
    audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    assert sr == 24000
    assert audio.dtype == np.float32
    assert audio.ndim == 1            # mono downmix
    assert len(audio) == 24000


def test_tts_forwards_reference_clip_and_prompt_text():
    fake = _FakeF5Model()
    c = _client(fake)
    c.post("/load_checkpoint", json={"checkpoint_dir": "runs/f5"})
    c.post("/tts", json={"text": "hi", "reference_clip": "ref.wav",
                         "prompt_text": "the transcript"})
    call = fake.tts_calls[-1]
    assert call["reference_clip"] == "ref.wav"
    assert call["prompt_text"] == "the transcript"     # F5 needs ref clip + its transcript


def test_tts_downmixes_stereo_to_mono():
    class _StereoModel(_FakeF5Model):
        def tts(self, text, reference_clip, prompt_text="", **kw):
            return np.zeros((24000, 2), dtype=np.float32), self.sr

    fake = _StereoModel()
    c = _client(fake)
    c.post("/load_checkpoint", json={"checkpoint_dir": "runs/f5"})
    r = c.post("/tts", json={"text": "hi", "reference_clip": "ref.wav", "prompt_text": "t"})
    assert r.status_code == 200
    audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    assert sr == 24000
    assert audio.ndim == 1
    assert len(audio) == 24000


# --- real model factory: ckpt-dir -> concrete ckpt FILE resolution (GPU-free) ---
#
# F5TTS(ckpt_file=...) needs a FILE path (load_checkpoint does ckpt_path.split(".")[-1]
# then load_file()/torch.load() — a directory raises). The real factory must resolve a
# concrete checkpoint file inside the given dir before constructing F5TTS. We inject a
# fake F5TTS constructor (f5_factory) that captures the ckpt_file/vocab_file it receives,
# so the resolution path is exercised without importing the real f5_tts package.


def _captured_factory():
    """Build the real _F5Model with a fake F5TTS constructor that records its kwargs."""
    from voxclone.serve.f5_server import _default_model_factory

    captured = {}

    def fake_f5(model=None, ckpt_file=None, vocab_file=None):
        captured["model"] = model
        captured["ckpt_file"] = ckpt_file
        captured["vocab_file"] = vocab_file
        return object()  # opaque model handle; only /tts touches it

    model = _default_model_factory(f5_factory=fake_f5)
    return model, captured


def test_load_checkpoint_resolves_model_last_pt_in_dir(tmp_path):
    ckpt = tmp_path / "model_last.pt"
    ckpt.write_bytes(b"")
    model, captured = _captured_factory()

    model.load_checkpoint(str(tmp_path), vocab_file=None)

    # F5TTS must receive the FILE, not the directory.
    assert captured["ckpt_file"] == str(ckpt)
    assert captured["ckpt_file"] != str(tmp_path)


def test_load_checkpoint_picks_newest_model_pt_when_no_model_last(tmp_path):
    import os

    older = tmp_path / "model_100.pt"
    newer = tmp_path / "model_500.pt"
    older.write_bytes(b"")
    newer.write_bytes(b"")
    # Make `newer` unambiguously newer by mtime.
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    model, captured = _captured_factory()

    model.load_checkpoint(str(tmp_path), vocab_file=None)

    assert captured["ckpt_file"] == str(newer)


def test_load_checkpoint_resolves_safetensors_when_present(tmp_path):
    ckpt = tmp_path / "model_1200.safetensors"
    ckpt.write_bytes(b"")
    model, captured = _captured_factory()

    model.load_checkpoint(str(tmp_path), vocab_file=None)

    assert captured["ckpt_file"] == str(ckpt)


def test_load_checkpoint_uses_file_path_as_is(tmp_path):
    ckpt = tmp_path / "some_ckpt.pt"
    ckpt.write_bytes(b"")
    model, captured = _captured_factory()

    model.load_checkpoint(str(ckpt), vocab_file=None)

    assert captured["ckpt_file"] == str(ckpt)


def test_load_checkpoint_picks_up_vocab_txt_in_dir(tmp_path):
    (tmp_path / "model_last.pt").write_bytes(b"")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("a\nb\n")
    model, captured = _captured_factory()

    model.load_checkpoint(str(tmp_path), vocab_file=None)

    assert captured["vocab_file"] == str(vocab)


def test_load_checkpoint_honours_explicit_vocab_file(tmp_path):
    (tmp_path / "model_last.pt").write_bytes(b"")
    (tmp_path / "vocab.txt").write_text("a\n")  # present, but explicit wins
    explicit = tmp_path / "custom_vocab.txt"
    explicit.write_text("x\n")
    model, captured = _captured_factory()

    model.load_checkpoint(str(tmp_path), vocab_file=str(explicit))

    assert captured["vocab_file"] == str(explicit)


def test_load_checkpoint_vocab_none_when_absent(tmp_path):
    (tmp_path / "model_last.pt").write_bytes(b"")  # no vocab.txt
    model, captured = _captured_factory()

    model.load_checkpoint(str(tmp_path), vocab_file=None)

    assert captured["vocab_file"] is None
