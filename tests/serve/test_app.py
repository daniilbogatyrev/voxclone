import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from voxclone.serve.app import create_app, create_studio_app

class FakeSynth:
    def synthesize(self, text, reference_clip, params):
        return np.zeros(2400, dtype=np.float32), 24000

def test_synthesize_returns_wav():
    app = create_app(synth_provider=lambda name: FakeSynth(), reference_clip="ref.wav")
    client = TestClient(app)
    resp = client.post("/synthesize", json={"text": "hello", "model": "gptsovits"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(resp.content))
    assert sr == 24000
    assert len(data) == 2400

def test_synthesize_rejects_empty_text():
    app = create_app(synth_provider=lambda name: FakeSynth(), reference_clip="ref.wav")
    client = TestClient(app)
    resp = client.post("/synthesize", json={"text": "", "model": "gptsovits"})
    assert resp.status_code == 422

def test_synthesize_unknown_model_returns_404():
    def provider(name):
        raise KeyError(name)
    app = create_app(provider, "ref.wav")
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/synthesize", json={"text": "hello", "model": "unknown"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# create_studio_app: the studio-backed live demo (GPU-free via an injected fake)
# --------------------------------------------------------------------------- #

class _Voice:
    def __init__(self, key, label):
        self.key, self.label = key, label


class FakeStudio:
    """Mimics the bits of voxclone.clone.studio that create_studio_app touches."""
    DEFAULT_VOICE = "f5_finetuned"
    DEFAULT_GERMAN_VOICE = "f5_zeroshot"
    _ALL = [_Voice("f5_finetuned", "F5 ft"), _Voice("f5_zeroshot", "F5 zs"),
            _Voice("chatterbox", "Chatterbox")]

    def __init__(self):
        self.calls = []

    def voices_for(self, language):
        # mirror the real studio: f5_finetuned is English-only
        return [v for v in self._ALL if not (language == "de" and v.key == "f5_finetuned")]

    def say(self, voice, text, language="en"):
        self.calls.append((voice, text, language))
        if voice not in {v.key for v in self._ALL}:
            raise KeyError(voice)
        if voice == "f5_finetuned" and language != "en":
            raise ValueError("English-only voice")
        return np.zeros(2400, dtype=np.float32), 24000, "out.wav"

    def restart(self, voice=None):
        self.calls.append(("restart", voice))


def _studio_client(studio):
    # frontend_dir absent -> skip the static mount, isolate the API from cwd
    return TestClient(create_studio_app(studio=studio, frontend_dir="does_not_exist"),
                      raise_server_exceptions=False)


def test_studio_voices_filtered_by_language():
    client = _studio_client(FakeStudio())
    en = client.get("/voices?language=en").json()
    assert [v["key"] for v in en["voices"]] == ["f5_finetuned", "f5_zeroshot", "chatterbox"]
    assert en["default"] == "f5_finetuned"
    de = client.get("/voices?language=de").json()
    assert "f5_finetuned" not in [v["key"] for v in de["voices"]]
    assert de["default"] == "f5_zeroshot"


def test_studio_clone_returns_wav_and_calls_studio():
    studio = FakeStudio()
    resp = _studio_client(studio).post(
        "/clone", json={"voice": "f5_zeroshot", "text": "hi", "language": "en"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(resp.content))
    assert sr == 24000 and len(data) == 2400
    assert studio.calls[-1] == ("f5_zeroshot", "hi", "en")


def test_studio_clone_empty_text_is_422():
    resp = _studio_client(FakeStudio()).post("/clone", json={"voice": "f5_zeroshot", "text": "  "})
    assert resp.status_code == 422


def test_studio_clone_wrong_language_is_400():
    resp = _studio_client(FakeStudio()).post(
        "/clone", json={"voice": "f5_finetuned", "text": "hi", "language": "de"})
    assert resp.status_code == 400


def test_studio_clone_unknown_voice_is_404():
    resp = _studio_client(FakeStudio()).post("/clone", json={"voice": "nope", "text": "hi"})
    assert resp.status_code == 404


def test_studio_restart_calls_studio():
    studio = FakeStudio()
    resp = _studio_client(studio).post("/restart", json={"voice": "f5_zeroshot"})
    assert resp.status_code == 200
    assert ("restart", "f5_zeroshot") in studio.calls


def test_studio_allows_cross_origin():
    # The presentation deck (voxclone.html) is opened from a different origin
    # (file:// or its own http.server port) and must be able to fetch /voices,
    # /languages and POST /clone -- so the studio app sends CORS headers.
    client = _studio_client(FakeStudio())
    resp = client.get("/voices?language=en", headers={"Origin": "http://localhost:8080"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
    # Preflight for the POST /clone the deck makes.
    pre = client.options(
        "/clone",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert pre.status_code == 200
    assert pre.headers.get("access-control-allow-origin") == "*"
