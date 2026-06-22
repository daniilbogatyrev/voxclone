"""GPU-free tests for voxclone.clone.studio.

Pure helpers are tested directly; the worker orchestration (launch-reuse, HTTP,
WAV decode, save) is exercised end-to-end against a threaded stdlib HTTP server
that impersonates engine_server — no conda env, no torch, no GPU.
"""
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from voxclone.clone import studio
from voxclone.clone.studio import (
    DEFAULT_GERMAN_VOICE,
    DEFAULT_VOICE,
    ENV_PY,
    REF_CLIP,
    REF_TEXT,
    VOICES,
    VOICES_BY_KEY,
    Voice,
    VoiceStudio,
    load_request,
    readiness_kind,
    synth_request,
    voices_for,
    worker_argv,
    worker_env,
)


# --------------------------------------------------------------------------- #
# Voice table integrity                                                       #
# --------------------------------------------------------------------------- #

def test_five_voices_unique_keys_and_ports():
    assert len(VOICES) == 5
    keys = [v.key for v in VOICES]
    ports = [v.port for v in VOICES]
    assert len(set(keys)) == 5
    assert len(set(ports)) == 5  # every voice gets its own port (no 9881/9882 clash)


def test_default_voice_exists_and_is_finetuned():
    assert DEFAULT_VOICE in VOICES_BY_KEY
    assert VOICES_BY_KEY[DEFAULT_VOICE].kind == "f5_server"


def test_every_voice_env_is_known():
    for v in VOICES:
        assert v.env in ENV_PY


def test_engine_server_voices_have_engine_ft_have_checkpoint():
    for v in VOICES:
        if v.kind == "engine_server":
            assert v.engine in {"f5", "xtts", "chatterbox"}
            assert v.checkpoint_dir is None
        else:
            assert v.checkpoint_dir is not None
            assert v.engine is None


# --------------------------------------------------------------------------- #
# worker_argv / worker_env                                                    #
# --------------------------------------------------------------------------- #

def test_worker_argv_engine_server_includes_engine_flag():
    v = VOICES_BY_KEY["f5_zeroshot"]
    argv = worker_argv(v)
    assert argv[0] == ENV_PY["f5tts"]
    assert argv[1:] == ["-m", "voxclone.serve.engine_server", "--port", "9882",
                        "--engine", "f5"]


def test_worker_argv_finetuned_has_no_engine_flag():
    v = VOICES_BY_KEY["f5_finetuned"]
    argv = worker_argv(v)
    assert "--engine" not in argv
    assert argv == [ENV_PY["f5tts"], "-m", "voxclone.serve.f5_server", "--port", "9892"]


def test_worker_env_pins_pythonpath_npp_and_offline():
    v = VOICES_BY_KEY["f5_zeroshot"]
    base = {"LD_LIBRARY_PATH": "/pre/existing", "CONDA_PREFIX": "/should/be/dropped"}
    ev = worker_env(v, base_env=base)
    assert ev["PYTHONPATH"].endswith("/src")
    assert ev["LD_LIBRARY_PATH"].startswith(studio.NPP["f5tts"])
    assert ev["LD_LIBRARY_PATH"].endswith(":/pre/existing")  # NPP prepended, existing kept
    assert ev["HF_HUB_OFFLINE"] == "1"
    assert ev["TRANSFORMERS_OFFLINE"] == "1"
    assert "CONDA_PREFIX" not in ev


def test_worker_env_neutralizes_kernel_leak_vars():
    # The Jupyter kernel exports an inline MPLBACKEND and the .venv exports
    # VIRTUAL_ENV/PYTHONHOME; these break a different-env worker if inherited.
    v = VOICES_BY_KEY["f5_zeroshot"]
    base = {
        "MPLBACKEND": "module://matplotlib_inline.backend_inline",
        "VIRTUAL_ENV": "/home/x/.venv",
        "PYTHONHOME": "/home/x/.venv",
    }
    ev = worker_env(v, base_env=base)
    assert ev["MPLBACKEND"] == "Agg"
    assert "VIRTUAL_ENV" not in ev
    assert "PYTHONHOME" not in ev


def test_worker_env_empty_ld_has_no_trailing_colon():
    v = VOICES_BY_KEY["f5_zeroshot"]
    ev = worker_env(v, base_env={})
    assert ev["LD_LIBRARY_PATH"] == studio.NPP["f5tts"]  # no leading/trailing ":"


def test_worker_env_coqui_only_for_xtts():
    assert "COQUI_TOS_AGREED" in worker_env(VOICES_BY_KEY["xtts_zeroshot"], base_env={})
    assert "COQUI_TOS_AGREED" in worker_env(VOICES_BY_KEY["xtts_finetuned"], base_env={})
    assert "COQUI_TOS_AGREED" not in worker_env(VOICES_BY_KEY["f5_zeroshot"], base_env={})
    assert "COQUI_TOS_AGREED" not in worker_env(VOICES_BY_KEY["chatterbox"], base_env={})


# --------------------------------------------------------------------------- #
# request builders                                                            #
# --------------------------------------------------------------------------- #

def test_synth_request_engine_server_f5_sends_ref_text():
    suffix, body = synth_request(VOICES_BY_KEY["f5_zeroshot"], "hello world")
    assert suffix == "/synth"
    assert body == {"text": "hello world", "ref_path": REF_CLIP,
                    "ref_text": REF_TEXT, "params": {"language": "en"}}


def test_synth_request_engine_server_xtts_chatterbox_empty_ref_text():
    for key in ("xtts_zeroshot", "chatterbox"):
        _, body = synth_request(VOICES_BY_KEY[key], "hi")
        assert body["ref_text"] == ""
        assert body["ref_path"] == REF_CLIP


def test_synth_request_f5_finetuned_uses_tts_fieldnames():
    suffix, body = synth_request(VOICES_BY_KEY["f5_finetuned"], "a sentence")
    assert suffix == "/tts"
    assert body == {"text": "a sentence", "reference_clip": REF_CLIP,
                    "prompt_text": REF_TEXT}
    assert "ref_path" not in body and "ref_text" not in body


def test_synth_request_xtts_finetuned_uses_language():
    suffix, body = synth_request(VOICES_BY_KEY["xtts_finetuned"], "a sentence")
    assert suffix == "/tts"
    assert body["reference_clip"] == REF_CLIP
    assert body["language"] == "en"


def test_load_request_none_for_zeroshot_set_for_finetuned():
    assert load_request(VOICES_BY_KEY["f5_zeroshot"]) is None
    assert load_request(VOICES_BY_KEY["chatterbox"]) is None
    suffix, body = load_request(VOICES_BY_KEY["f5_finetuned"])
    assert suffix == "/load_checkpoint"
    assert body["checkpoint_dir"] == VOICES_BY_KEY["f5_finetuned"].checkpoint_dir
    _, xbody = load_request(VOICES_BY_KEY["xtts_finetuned"])
    assert xbody["checkpoint_dir"] == VOICES_BY_KEY["xtts_finetuned"].checkpoint_dir


def test_readiness_kind_is_http_for_all_voices():
    # Every studio server (incl. the fine-tuned ones) now exposes GET /health,
    # so readiness is identity-aware for all five voices.
    for v in VOICES:
        assert readiness_kind(v) == "http_health"


def test_pid_is_our_worker_rejects_missing_and_foreign_pids():
    import os
    from voxclone.clone.studio import _pid_is_our_worker
    assert _pid_is_our_worker(2 ** 31, 9892) is False          # no such /proc/<pid>
    assert _pid_is_our_worker(os.getpid(), 9892) is False       # our cmdline is pytest, not a worker


# --------------------------------------------------------------------------- #
# save naming                                                                 #
# --------------------------------------------------------------------------- #

def test_save_increments_and_writes(tmp_path):
    st = VoiceStudio(outdir=str(tmp_path), on_status=lambda m: None)
    audio = np.zeros(2400, dtype="float32")
    p1 = st._save(VOICES_BY_KEY["f5_finetuned"], audio, 24000)
    p2 = st._save(VOICES_BY_KEY["f5_finetuned"], audio, 24000)
    assert p1.name == "f5_finetuned_001.wav"
    assert p2.name == "f5_finetuned_002.wav"
    assert p1.exists() and p2.exists()


# --------------------------------------------------------------------------- #
# end-to-end say() against a fake engine_server (threaded, no GPU)            #
# --------------------------------------------------------------------------- #

class _FakeEngineHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path == "/health":
            payload = json.dumps({"engine": "fake", "ready": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)  # consume body
        buf = io.BytesIO()
        audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(2400) / 24000)).astype("float32")
        sf.write(buf, audio, 24000, format="WAV", subtype="FLOAT")
        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _FakeProc:
    pid = 999999

    def poll(self):
        return None  # always "alive"


@pytest.fixture
def fake_engine():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = ThreadingHTTPServer(("127.0.0.1", port), _FakeEngineHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()


def test_say_end_to_end_against_fake(monkeypatch, tmp_path, fake_engine):
    # A fake engine_server voice pointed at the running threaded server.
    fake_voice = Voice("fake", "Fake", env="f5tts", kind="engine_server",
                       module="voxclone.serve.engine_server", port=fake_engine,
                       engine="f5", ref_text="")
    monkeypatch.setitem(VOICES_BY_KEY, "fake", fake_voice)
    # Don't spawn a real subprocess — the server is already up in a thread.
    monkeypatch.setattr(VoiceStudio, "_launch",
                        lambda self, v: setattr(self, "_proc", _FakeProc()))
    monkeypatch.setattr(VoiceStudio, "_reclaim_port", lambda self, port: None)

    st = VoiceStudio(outdir=str(tmp_path), on_status=lambda m: None)
    wav, sr, path = st.say("fake", "hello there")

    assert sr == 24000
    assert isinstance(wav, np.ndarray) and wav.dtype == np.float32
    assert wav.shape[0] == 2400
    assert path.exists() and path.suffix == ".wav"


def test_say_rejects_empty_text(tmp_path):
    st = VoiceStudio(outdir=str(tmp_path), on_status=lambda m: None)
    with pytest.raises(ValueError):
        st.say(DEFAULT_VOICE, "   ")


# --------------------------------------------------------------------------- #
# language / German                                                           #
# --------------------------------------------------------------------------- #

def test_voice_langs_f5_finetuned_english_only_rest_speak_german():
    assert VOICES_BY_KEY["f5_finetuned"].langs == ("en",)
    for key in ("f5_zeroshot", "xtts_finetuned", "xtts_zeroshot", "chatterbox"):
        assert "de" in VOICES_BY_KEY[key].langs


def test_voices_for_filters_by_language():
    assert {v.key for v in voices_for("en")} == {v.key for v in VOICES}      # all 5
    assert {v.key for v in voices_for("de")} == {
        "f5_zeroshot", "xtts_finetuned", "xtts_zeroshot", "chatterbox"}      # not f5_finetuned


def test_default_german_voice_speaks_german():
    assert DEFAULT_GERMAN_VOICE in VOICES_BY_KEY
    assert "de" in VOICES_BY_KEY[DEFAULT_GERMAN_VOICE].langs


def test_synth_request_threads_language():
    # engine_server: language rides in params
    _, body = synth_request(VOICES_BY_KEY["f5_zeroshot"], "hallo", language="de")
    assert body["params"] == {"language": "de"}
    # xtts fine-tuned server: language is a top-level field
    _, xbody = synth_request(VOICES_BY_KEY["xtts_finetuned"], "hallo", language="de")
    assert xbody["language"] == "de"


def test_say_rejects_unsupported_language_before_launch(tmp_path):
    # f5_finetuned is English-only; asking for German must fail fast (no worker launch).
    st = VoiceStudio(outdir=str(tmp_path), on_status=lambda m: None)
    with pytest.raises(ValueError, match="English-only"):
        st.say("f5_finetuned", "Guten Tag", language="de")


# --------------------------------------------------------------------------- #
# custom reference — "clone your own voice" (zero-shot only)                   #
# --------------------------------------------------------------------------- #

def test_synth_request_custom_reference_overrides_speaker():
    # A zero-shot engine_server voice points at ANY clip + its transcript.
    suffix, body = synth_request(VOICES_BY_KEY["f5_zeroshot"], "hi",
                                 ref_path="/tmp/me.wav", ref_text="my own transcript")
    assert suffix == "/synth"
    assert body["ref_path"] == "/tmp/me.wav"
    assert body["ref_text"] == "my own transcript"


def test_synth_request_custom_ref_text_empty_string_is_respected():
    # An explicit "" must NOT fall back to the voice's Daniil transcript.
    _, body = synth_request(VOICES_BY_KEY["f5_zeroshot"], "hi",
                            ref_path="/tmp/me.wav", ref_text="")
    assert body["ref_text"] == ""


def test_synth_request_defaults_to_daniil_when_no_ref():
    _, body = synth_request(VOICES_BY_KEY["f5_zeroshot"], "hi")
    assert body["ref_path"] == REF_CLIP
    assert body["ref_text"] == REF_TEXT


def test_say_rejects_custom_reference_for_finetuned_voices(tmp_path):
    # The fine-tuned voices ARE Daniil (baked into weights) — a custom clip is meaningless.
    st = VoiceStudio(outdir=str(tmp_path), on_status=lambda m: None)
    for key in ("f5_finetuned", "xtts_finetuned"):
        with pytest.raises(ValueError, match="zero-shot"):
            st.say(key, "hi", ref_path="/tmp/me.wav")


def test_say_threads_custom_reference_onto_the_wire(monkeypatch, tmp_path):
    # The custom ref_path/ref_text must reach the POST body for a zero-shot voice.
    st = VoiceStudio(outdir=str(tmp_path), on_status=lambda m: None)
    monkeypatch.setattr(st, "_ensure_worker", lambda voice, timeout: None)
    captured = {}

    def fake_post(url, body, timeout):
        captured["url"], captured["body"] = url, body
        buf = io.BytesIO()
        sf.write(buf, np.zeros(2400, dtype="float32"), 24000, format="WAV", subtype="FLOAT")
        return buf.getvalue()

    monkeypatch.setattr(studio, "_post_bytes", fake_post)
    wav, sr, path = st.say("f5_zeroshot", "hallo", language="de",
                           ref_path="/my/voice.wav", ref_text="this is me speaking")
    assert captured["body"]["ref_path"] == "/my/voice.wav"
    assert captured["body"]["ref_text"] == "this is me speaking"
    assert captured["body"]["params"] == {"language": "de"}
    assert sr == 24000 and path.exists()
