import io

import numpy as np
import soundfile as sf

from voxclone.synth import client


def _wav_bytes(n=3000, sr=24000):
    buf = io.BytesIO()
    sf.write(buf, (0.1 * np.ones(n)).astype("float32"), sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def test_engines_map_has_four():
    assert set(client.ENGINES) == {"gptsovits_v2pro", "xtts_v2", "f5_tts", "chatterbox"}
    assert client.ENGINES["f5_tts"]["port"] == 9882


def test_synth_engine_server_kind(monkeypatch):
    from voxclone.serve.engine_server import SynthReq

    class FakeResp:
        content = _wav_bytes()

        def raise_for_status(self):
            pass

    captured = {}

    def fake_post(url, json=None, **k):
        captured["body"] = json
        return FakeResp()

    monkeypatch.setattr(client.httpx, "post", fake_post)
    audio, sr = client.synth("xtts_v2", "hello", "/ref.wav", ref_text="r",
                             params={"speed": 1.5})
    assert sr == 24000 and audio.dtype == np.float32 and len(audio) == 3000

    # The request body's keys must match engine_server's SynthReq fields exactly,
    # so a field-name drift between client.synth and SynthReq is caught here.
    body = captured["body"]
    assert set(body) == set(SynthReq.model_fields) == {"text", "ref_path", "ref_text", "params"}
    assert body == {
        "text": "hello",
        "ref_path": "/ref.wav",
        "ref_text": "r",
        "params": {"speed": 1.5},
    }
    # The server must be able to parse exactly this body without drift.
    parsed = SynthReq(**body)
    assert parsed.text == "hello" and parsed.ref_path == "/ref.wav"
    assert parsed.ref_text == "r" and parsed.params == {"speed": 1.5}


def test_synth_api_v2_kind(monkeypatch):
    calls = {}

    class FakeGS:
        def __init__(self, checkpoint):
            calls["ckpt"] = checkpoint

        def synthesize(self, text, ref, params):
            calls["params"] = params
            return np.zeros(100, dtype="float32"), 32000

    monkeypatch.setattr(client, "GPTSoVITSSynth", FakeGS)
    audio, sr = client.synth("gptsovits_v2pro", "hi", "/ref.wav", ref_text="the transcript")
    assert sr == 32000
    assert calls["params"]["prompt_text"] == "the transcript"  # ref_text -> prompt_text


def test_health_false_when_down(monkeypatch):
    def boom(*a, **k):
        raise client.httpx.ConnectError("down")

    monkeypatch.setattr(client.httpx, "get", boom)
    assert client.health("xtts_v2") is False
