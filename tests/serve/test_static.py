from fastapi.testclient import TestClient
from voxclone.serve.app import create_app

class FakeSynth:
    def synthesize(self, text, reference_clip, params):
        import numpy as np
        return np.zeros(10, dtype=np.float32), 24000

def test_index_served():
    app = create_app(lambda n: FakeSynth(), "ref.wav", frontend_dir="frontend")
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "voxclone" in resp.text.lower()
