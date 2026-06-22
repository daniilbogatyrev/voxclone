"""P07-T7: XTTSSynth / F5Synth checkpoint-bound adapters (eval/serve-facing SynthAdapter).

These adapters are the eval/serve consumers of the SynthAdapter protocol
(`synth.synthesize(text, reference_clip, params) -> (np.ndarray, int)`), DISTINCT
from the notebook-facing XTTSAdapter / F5Adapter (`synthesize(text, ref_path,
ref_text="", params=None)`). They mirror GPTSoVITSSynth: the checkpoint is bound at
construction and applied SERVER-SIDE by the adapter itself, because eval/runner.py
calls `synth.synthesize(text, reference_clip, {})` with EMPTY params.

The regression guarded against here is the latent GPT-SoVITS bug where the finetuned
weights are read from `params['gpt_weights']` -- which eval never sends -- so a
checkpoint-bound adapter that only switched weights on non-empty params would silently
synthesize from the BASE model. XTTSSynth/F5Synth must `/load_checkpoint` from
`self.checkpoint` even when params is `{}`.
"""
import io

import numpy as np
import soundfile as sf

from voxclone.synth.xtts import XTTSSynth
from voxclone.synth.f5 import F5Synth


# ---------------------------------------------------------------------------
# XTTSSynth
# ---------------------------------------------------------------------------

def test_xtts_synthesize_empty_params_via_seam_returns_float32_24k():
    """eval calls synthesize(text, ref, {}); the seam receives the construction
    checkpoint (NOT a param) and the adapter coerces float32 mono + int sr."""
    seen = {}

    def fake_generate(text, reference_clip, checkpoint, params):
        seen.update(text=text, ref=reference_clip, ckpt=checkpoint, params=params)
        return np.zeros(24000, dtype=np.float32), 24000

    synth = XTTSSynth(checkpoint="runs/xtts", generate_fn=fake_generate)
    audio, sr = synth.synthesize("hello", "ref.wav", {})  # EMPTY params, as eval calls it
    assert sr == 24000
    assert audio.dtype == np.float32
    assert audio.ndim == 1 and len(audio) == 24000
    # The checkpoint is construction-bound, passed to the seam itself -- not pulled
    # from params (params is empty here).
    assert seen["ckpt"] == "runs/xtts"
    assert seen["ref"] == "ref.wav"
    assert seen["params"] == {}


def test_xtts_synthesize_coerces_list_audio_and_float_sr():
    def fake_generate(text, reference_clip, checkpoint, params):
        return [0.0, 1.0, -1.0], 24000.0  # list + float sr

    synth = XTTSSynth(checkpoint="runs/xtts", generate_fn=fake_generate)
    audio, sr = synth.synthesize("hi", "ref.wav", {})
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
    assert isinstance(sr, int) and sr == 24000


def test_xtts_real_generate_loads_checkpoint_then_posts_tts_with_empty_params(monkeypatch):
    """The REAL generate path must POST /load_checkpoint from the construction
    checkpoint BEFORE /tts, even with empty params -- this is the anti-regression
    guard against the GPTSoVITS params['gpt_weights'] bug."""
    from voxclone.synth import xtts as x

    buf = io.BytesIO()
    sf.write(buf, np.zeros(12000, dtype=np.float32), 24000, format="WAV")
    wav_bytes = buf.getvalue()
    posts = []

    class FakeResp:
        content = wav_bytes

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        posts.append((url, json))
        return FakeResp()

    monkeypatch.setattr(x.httpx, "post", fake_post)

    # EMPTY params except the server_url override -- mirrors the eval path closely
    # (the adapter binds checkpoint at construction, not via params).
    audio, sr = x._real_generate_xtts(
        "hi", "ref.wav", "runs/xtts", {"server_url": "http://127.0.0.1:9881"}
    )
    assert sr == 24000
    assert audio.dtype == np.float32
    assert audio.ndim == 1 and len(audio) == 12000

    urls = [u for u, _ in posts]
    assert any(u.endswith("/load_checkpoint") for u in urls), "must /load_checkpoint"
    assert any(u.endswith("/tts") for u in urls), "must /tts"
    # /load_checkpoint precedes /tts
    assert urls.index(next(u for u in urls if u.endswith("/load_checkpoint"))) < urls.index(
        next(u for u in urls if u.endswith("/tts"))
    )
    # checkpoint_dir comes from the construction checkpoint, NOT params['gpt_weights']
    load_body = next(b for u, b in posts if u.endswith("/load_checkpoint"))
    assert load_body["checkpoint_dir"] == "runs/xtts"
    tts_body = next(b for u, b in posts if u.endswith("/tts"))
    assert tts_body["text"] == "hi"
    assert tts_body["reference_clip"] == "ref.wav"
    # MUST NOT read finetuned weights out of params (the GPTSoVITS latent bug).
    assert "gpt_weights" not in load_body


def test_xtts_real_generate_downmixes_stereo_to_mono(monkeypatch):
    """A 2-channel WAV from the server is decoded and downmixed to mono. This path
    (sf.read + the audio.ndim > 1 mean) runs entirely under the mocked httpx.post --
    so it is covered, not server-only."""
    from voxclone.synth import xtts as x

    n_frames = 10000
    left = np.full(n_frames, 0.5, dtype=np.float32)
    right = np.full(n_frames, -0.1, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)  # shape (n_frames, 2)
    buf = io.BytesIO()
    # FLOAT subtype so the 2-channel float content round-trips without 16-bit PCM
    # quantization -- lets us assert the exact per-frame channel mean.
    sf.write(buf, stereo, 24000, format="WAV", subtype="FLOAT")
    wav_bytes = buf.getvalue()

    class FakeResp:
        content = wav_bytes

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        x.httpx, "post", lambda url, json=None, timeout=None: FakeResp()
    )
    audio, sr = x._real_generate_xtts(
        "hi", "ref.wav", "runs/xtts", {"server_url": "http://127.0.0.1:9881"}
    )
    assert sr == 24000
    assert audio.dtype == np.float32
    assert audio.ndim == 1  # stereo was downmixed to mono
    assert len(audio) == n_frames  # frame length preserved
    # mean of the two channels, per frame
    np.testing.assert_allclose(audio, (left + right) / 2.0, rtol=0, atol=1e-6)


def test_xtts_end_to_end_checkpoint_binding_through_synthesize(monkeypatch):
    """Full path: XTTSSynth(checkpoint=...).synthesize(text, ref, {}) drives the
    server to /load_checkpoint from self.checkpoint then /tts."""
    from voxclone.synth import xtts as x

    buf = io.BytesIO()
    sf.write(buf, np.zeros(6000, dtype=np.float32), 24000, format="WAV")
    wav_bytes = buf.getvalue()
    posts = []

    class FakeResp:
        content = wav_bytes

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        x.httpx, "post", lambda url, json=None, timeout=None: posts.append((url, json)) or FakeResp()
    )
    synth = XTTSSynth(checkpoint="runs/xtts")  # default generate_fn = real path
    audio, sr = synth.synthesize("sentence", "enroll.wav", {})  # EMPTY params
    assert sr == 24000 and audio.dtype == np.float32 and len(audio) == 6000
    load_body = next(b for u, b in posts if u.endswith("/load_checkpoint"))
    assert load_body["checkpoint_dir"] == "runs/xtts"


# ---------------------------------------------------------------------------
# F5Synth
# ---------------------------------------------------------------------------

def test_f5_synthesize_empty_params_via_seam_returns_float32_24k():
    seen = {}

    def fake_generate(text, reference_clip, checkpoint, params):
        seen.update(ckpt=checkpoint, params=params)
        return np.zeros(24000, dtype=np.float32), 24000

    synth = F5Synth(checkpoint="runs/f5", generate_fn=fake_generate)
    audio, sr = synth.synthesize("hello", "ref.wav", {})
    assert sr == 24000
    assert audio.dtype == np.float32
    assert audio.ndim == 1 and len(audio) == 24000
    assert seen["ckpt"] == "runs/f5"
    assert seen["params"] == {}


def test_f5_real_generate_loads_checkpoint_then_posts_tts_with_prompt_text(monkeypatch):
    from voxclone.synth import f5 as f

    buf = io.BytesIO()
    sf.write(buf, np.zeros(9000, dtype=np.float32), 24000, format="WAV")
    wav_bytes = buf.getvalue()
    posts = []

    class FakeResp:
        content = wav_bytes

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        f.httpx, "post", lambda url, json=None, timeout=None: posts.append((url, json)) or FakeResp()
    )
    audio, sr = f._real_generate_f5(
        "hi", "ref.wav", "runs/f5",
        {"prompt_text": "the transcript", "server_url": "http://127.0.0.1:9882"},
    )
    assert sr == 24000
    assert audio.dtype == np.float32
    assert audio.ndim == 1 and len(audio) == 9000

    urls = [u for u, _ in posts]
    assert any(u.endswith("/load_checkpoint") for u in urls)
    assert any(u.endswith("/tts") for u in urls)
    load_body = next(b for u, b in posts if u.endswith("/load_checkpoint"))
    assert load_body["checkpoint_dir"] == "runs/f5"  # construction-bound, not from params
    tts_body = next(b for u, b in posts if u.endswith("/tts"))
    # F5 conditions on the ref clip AND its transcript.
    assert tts_body["prompt_text"] == "the transcript"
    assert tts_body["reference_clip"] == "ref.wav"


def test_f5_real_generate_downmixes_stereo_to_mono(monkeypatch):
    """A 2-channel WAV from the f5_server is decoded and downmixed to mono. The
    sf.read decode + stereo->mono mean run under the mocked httpx.post, so they
    are covered rather than server-only."""
    from voxclone.synth import f5 as f

    n_frames = 7000
    left = np.full(n_frames, 0.25, dtype=np.float32)
    right = np.full(n_frames, 0.75, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)  # shape (n_frames, 2)
    buf = io.BytesIO()
    # FLOAT subtype so the 2-channel float content round-trips without 16-bit PCM
    # quantization -- lets us assert the exact per-frame channel mean.
    sf.write(buf, stereo, 24000, format="WAV", subtype="FLOAT")
    wav_bytes = buf.getvalue()

    class FakeResp:
        content = wav_bytes

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        f.httpx, "post", lambda url, json=None, timeout=None: FakeResp()
    )
    audio, sr = f._real_generate_f5(
        "hi", "ref.wav", "runs/f5",
        {"prompt_text": "t", "server_url": "http://127.0.0.1:9882"},
    )
    assert sr == 24000
    assert audio.dtype == np.float32
    assert audio.ndim == 1  # stereo was downmixed to mono
    assert len(audio) == n_frames  # frame length preserved
    np.testing.assert_allclose(audio, (left + right) / 2.0, rtol=0, atol=1e-6)


def test_f5_end_to_end_checkpoint_binding_with_empty_params(monkeypatch):
    """Empty-params eval path: F5Synth(checkpoint=...).synthesize(text, ref, {})
    still loads the checkpoint server-side (prompt_text defaults to empty)."""
    from voxclone.synth import f5 as f

    buf = io.BytesIO()
    sf.write(buf, np.zeros(3000, dtype=np.float32), 24000, format="WAV")
    wav_bytes = buf.getvalue()
    posts = []

    class FakeResp:
        content = wav_bytes

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        f.httpx, "post", lambda url, json=None, timeout=None: posts.append((url, json)) or FakeResp()
    )
    synth = F5Synth(checkpoint="runs/f5")
    audio, sr = synth.synthesize("sentence", "enroll.wav", {})  # EMPTY params
    assert sr == 24000 and audio.dtype == np.float32 and len(audio) == 3000
    load_body = next(b for u, b in posts if u.endswith("/load_checkpoint"))
    assert load_body["checkpoint_dir"] == "runs/f5"


# ---------------------------------------------------------------------------
# Both satisfy the SynthAdapter protocol shape
# ---------------------------------------------------------------------------

def test_both_satisfy_synth_adapter_signature():
    """checkpoint-bound ctor + 3-arg synthesize (text, reference_clip, params)."""
    for cls in (XTTSSynth, F5Synth):
        synth = cls(checkpoint="ckpt", generate_fn=lambda t, r, c, p: (np.zeros(8, np.float32), 24000))
        audio, sr = synth.synthesize("t", "r.wav", {})
        assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
        assert isinstance(sr, int)
