"""English footgun guard: the GPT-SoVITS default text_split_method must NOT be
'cut5' (cut5 splits on every comma and collapses English to silence). The safe
general-purpose English default is 'cut4'; callers may still override.
"""
import io
import numpy as np
import soundfile as sf
from voxclone.synth import gptsovits as g


def _fake_wav_bytes(n=4000, sr=32000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(n, dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def _capture_post(monkeypatch):
    """Patch httpx.post to capture the payload and return a decodable WAV."""
    wav_bytes = _fake_wav_bytes()
    captured = {}

    class FakeResp:
        content = wav_bytes

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr(g.httpx, "post", fake_post)
    return captured


def test_default_text_split_method_is_cut4_not_cut5(monkeypatch):
    captured = _capture_post(monkeypatch)
    # No text_split_method supplied -> must default to the English-safe 'cut4'.
    g._real_generate("Hello there, world. How are you?", "ref.wav", "/ckpt",
                     {"prompt_text": "hi"})
    assert captured["payload"]["text_split_method"] == "cut4"
    # And it must never silently fall back to the English-collapsing 'cut5'.
    assert captured["payload"]["text_split_method"] != "cut5"


def test_explicit_text_split_method_is_honored(monkeypatch):
    captured = _capture_post(monkeypatch)
    # An explicit single-sentence choice (cut0) must be passed through unchanged.
    g._real_generate("One single sentence.", "ref.wav", "/ckpt",
                     {"prompt_text": "hi", "text_split_method": "cut0"})
    assert captured["payload"]["text_split_method"] == "cut0"
