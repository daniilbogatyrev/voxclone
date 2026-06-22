import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from voxclone.serve import engine_server


def test_health_and_synth(monkeypatch):
    def fake(text, ref_path, ref_text, params):
        return (0.2 * np.ones(4096)).astype("float32"), 24000

    # build the app with a fake adapter (no model load)
    app = engine_server.make_app("xtts", adapter_factory=lambda: _FakeAdapter(fake))
    client = TestClient(app)

    h = client.get("/health").json()
    assert h["engine"] == "xtts" and h["ready"] is True

    r = client.post("/synth", json={"text": "hi", "ref_path": "/r.wav", "ref_text": "x"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/")
    audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    assert sr == 24000 and len(audio) == 4096


class _FakeAdapter:
    def __init__(self, fn):
        self._fn = fn

    def load(self):
        pass

    def synthesize(self, text, ref_path, ref_text="", params=None):
        return self._fn(text, ref_path, ref_text, params or {})
