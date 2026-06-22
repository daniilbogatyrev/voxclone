import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from voxclone.serve.xtts_server import make_app, DEFAULT_PORT, _resolve_xtts_checkpoint


class _FakeModel:
    """GPU-free stand-in for the XTTS model the real factory would build."""

    sr = 24000

    def __init__(self):
        self.loaded = None
        self.latent_calls = []

    def load_checkpoint(self, checkpoint_dir):
        self.loaded = checkpoint_dir

    def get_conditioning_latents(self, reference_clip):
        # The server caches this keyed by reference_clip; record every real call.
        self.latent_calls.append(reference_clip)
        return ("gpt_latent", "speaker_emb")

    def tts(self, text, reference_clip, latents=None, **kw):
        return np.zeros(self.sr, dtype=np.float32), self.sr


def _stereo_fake():
    fake = _FakeModel()

    def tts(text, reference_clip, latents=None, **kw):
        # 2-channel audio to exercise the mono downmix path.
        return np.zeros((fake.sr, 2), dtype=np.float32), fake.sr

    fake.tts = tts
    return fake


def test_distinct_port_is_9881():
    assert DEFAULT_PORT == 9881


def test_resolve_xtts_checkpoint_prefers_best_model(tmp_path):
    # Coqui trainer writes best_model.pth (no bare model.pth); resolver must find it.
    (tmp_path / "best_model.pth").write_bytes(b"x")
    (tmp_path / "checkpoint_1000.pth").write_bytes(b"x")
    assert _resolve_xtts_checkpoint(str(tmp_path)) == str(tmp_path / "best_model.pth")


def test_resolve_xtts_checkpoint_falls_back_to_model_then_newest(tmp_path):
    # No best_model.pth, but a plain model.pth -> use it.
    (tmp_path / "model.pth").write_bytes(b"x")
    assert _resolve_xtts_checkpoint(str(tmp_path)) == str(tmp_path / "model.pth")


def test_resolve_xtts_checkpoint_prefers_best_eval_over_last_step(tmp_path):
    # No bare best_model.pth: best_model_<step> must win over a newer checkpoint_<step>.
    (tmp_path / "best_model_936.pth").write_bytes(b"x")
    (tmp_path / "checkpoint_1000.pth").write_bytes(b"x")
    assert _resolve_xtts_checkpoint(str(tmp_path)) == str(tmp_path / "best_model_936.pth")


def test_resolve_xtts_checkpoint_picks_highest_checkpoint_step(tmp_path):
    (tmp_path / "checkpoint_900.pth").write_bytes(b"x")
    (tmp_path / "checkpoint_1000.pth").write_bytes(b"x")
    assert _resolve_xtts_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint_1000.pth")


def test_resolve_xtts_checkpoint_never_picks_auxiliary_weights(tmp_path):
    # dvae.pth / mel_stats.pth / speakers_xtts.pth must never be chosen.
    for aux in ("dvae.pth", "mel_stats.pth", "speakers_xtts.pth"):
        (tmp_path / aux).write_bytes(b"x")
    assert _resolve_xtts_checkpoint(str(tmp_path)) == str(tmp_path / "model.pth")  # default, not an aux file


def test_resolve_xtts_checkpoint_passthrough_file(tmp_path):
    f = tmp_path / "weights.pth"
    f.write_bytes(b"x")
    assert _resolve_xtts_checkpoint(str(f)) == str(f)


def test_resolve_xtts_checkpoint_default_when_empty(tmp_path):
    # Nothing on disk -> default to <dir>/model.pth (let Coqui surface the error).
    assert _resolve_xtts_checkpoint(str(tmp_path)) == str(tmp_path / "model.pth")


def test_health_endpoint():
    client = TestClient(make_app(model_factory=lambda: _FakeModel()))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_load_checkpoint_binds_server_side():
    fake = _FakeModel()
    client = TestClient(make_app(model_factory=lambda: fake))
    resp = client.post("/load_checkpoint", json={"checkpoint_dir": "runs/xtts"})
    assert resp.status_code == 200
    assert resp.json() == {"loaded": "runs/xtts"}
    # Checkpoint is bound on the server, not passed through /tts params.
    assert fake.loaded == "runs/xtts"


def test_round_trip_load_then_tts_returns_24k_float32_mono():
    fake = _FakeModel()
    client = TestClient(make_app(model_factory=lambda: fake))

    assert client.post("/load_checkpoint", json={"checkpoint_dir": "runs/xtts"}).status_code == 200
    assert fake.loaded == "runs/xtts"

    resp = client.post("/tts", json={"text": "hi", "reference_clip": "ref.wav"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/")

    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    assert sr == 24000
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == 24000


def test_tts_downmixes_stereo_to_mono():
    fake = _stereo_fake()
    client = TestClient(make_app(model_factory=lambda: fake))
    resp = client.post("/tts", json={"text": "hi", "reference_clip": "ref.wav"})
    assert resp.status_code == 200
    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    assert sr == 24000
    assert audio.ndim == 1
    assert len(audio) == 24000


def test_conditioning_latents_cached_by_reference_clip():
    fake = _FakeModel()
    client = TestClient(make_app(model_factory=lambda: fake))

    # Same reference clip twice -> latents computed once (cache hit on the second call).
    client.post("/tts", json={"text": "one", "reference_clip": "ref.wav"})
    client.post("/tts", json={"text": "two", "reference_clip": "ref.wav"})
    assert fake.latent_calls == ["ref.wav"]

    # A different reference clip is a distinct cache key -> recomputed.
    client.post("/tts", json={"text": "three", "reference_clip": "other.wav"})
    assert fake.latent_calls == ["ref.wav", "other.wav"]


def test_model_built_lazily_via_factory_only_once():
    builds = []

    def factory():
        builds.append(1)
        return _FakeModel()

    client = TestClient(make_app(model_factory=factory))
    client.post("/load_checkpoint", json={"checkpoint_dir": "runs/xtts"})
    client.post("/tts", json={"text": "hi", "reference_clip": "ref.wav"})
    client.post("/tts", json={"text": "bye", "reference_clip": "ref.wav"})
    assert len(builds) == 1
