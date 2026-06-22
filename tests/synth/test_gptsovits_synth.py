import numpy as np
from voxclone.synth.gptsovits import GPTSoVITSSynth

def test_synthesize_returns_audio_and_sr():
    def fake_generate(text, reference_clip, checkpoint, params):
        return np.zeros(24000, dtype=np.float32), 24000
    synth = GPTSoVITSSynth(checkpoint="/ckpt", generate_fn=fake_generate)
    audio, sr = synth.synthesize("hello", "ref.wav", {})
    assert sr == 24000
    assert audio.dtype == np.float32
    assert len(audio) == 24000

def test_real_generate_posts_to_tts_and_decodes_wav(monkeypatch):
    import io
    import numpy as np
    import soundfile as sf
    from voxclone.synth import gptsovits as g

    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 32000, format="WAV")
    wav_bytes = buf.getvalue()

    calls = {}

    class FakeResp:
        content = wav_bytes
        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        calls["url"] = url
        calls["payload"] = json
        return FakeResp()

    monkeypatch.setattr(g.httpx, "post", fake_post)
    audio, sr = g._real_generate("hello world", "ref.wav", "/ckpt",
                                 {"prompt_text": "hi", "server_url": "http://x:9880"})
    assert sr == 32000
    assert audio.dtype == np.float32
    assert len(audio) == 16000
    assert calls["url"] == "http://x:9880/tts"
    assert calls["payload"]["text"] == "hello world"
    assert calls["payload"]["ref_audio_path"] == "ref.wav"
    assert calls["payload"]["prompt_text"] == "hi"
    assert calls["payload"]["media_type"] == "wav"

def test_real_generate_switches_finetuned_weights(monkeypatch):
    import io
    import numpy as np
    import soundfile as sf
    from voxclone.synth import gptsovits as g

    buf = io.BytesIO()
    sf.write(buf, np.zeros(8000, dtype=np.float32), 32000, format="WAV")
    wav_bytes = buf.getvalue()
    gets = []

    class FakeResp:
        content = wav_bytes
        def raise_for_status(self):
            pass

    monkeypatch.setattr(g.httpx, "post", lambda *a, **k: FakeResp())
    def fake_get(url, params=None, timeout=None):
        gets.append((url, params))
        return FakeResp()
    monkeypatch.setattr(g.httpx, "get", fake_get)

    g._real_generate("hi", "ref.wav", "/ckpt",
                     {"prompt_text": "x", "gpt_weights": "/w/gpt.ckpt",
                      "sovits_weights": "/w/sovits.pth"})
    urls = [u for u, _ in gets]
    assert any(u.endswith("/set_gpt_weights") for u in urls)
    assert any(u.endswith("/set_sovits_weights") for u in urls)
    assert any(p == {"weights_path": "/w/gpt.ckpt"} for _, p in gets)
