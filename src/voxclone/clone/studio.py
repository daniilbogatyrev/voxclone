"""Zero-terminal voice-clone studio: a thin-kernel manager that launches one
engine worker at a time in its own conda env (no ``conda activate``), keeps it
warm, and returns Daniil's cloned voice for a typed sentence.

The notebook kernel (the project ``.venv``) imports ONLY this module — stdlib +
``numpy`` + ``soundfile``, never torch / f5_tts / TTS. The heavy TTS libraries
live in per-engine conda envs; we drive their existing FastAPI servers
(``engine_server`` / ``f5_server`` / ``xtts_server``) over localhost HTTP using
stdlib ``urllib``.
"""
from __future__ import annotations

import atexit
import glob
import io
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------- #
# Paths & constants (machine layout)                                          #
# --------------------------------------------------------------------------- #

ROOT = str(Path(__file__).resolve().parents[3])  # .../clone -> voxclone -> src -> repo root
CONDA_ENVS = Path.home() / "miniforge3" / "envs"

ENV_PY = {e: str(CONDA_ENVS / e / "bin" / "python") for e in ("f5tts", "xtts", "chatterbox")}
NPP = {
    e: str(CONDA_ENVS / e / "lib" / "python3.11" / "site-packages" / "nvidia" / "npp" / "lib")
    for e in ENV_PY
}

REF_CLIP = f"{ROOT}/experiments/danil/reference/danill_ref9.wav"
# Verbatim transcript of REF_CLIP (from experiments/ab_gen_f5_ft.py — the run
# that produced the bake-off #1). F5 conditions on both the clip and this text.
REF_TEXT = (
    "This is my natural speaking voice, calm, clear and steady. "
    "As I read these few lines aloud today, I speak at an even pace with"
)
F5_CKPT_DIR = f"{ROOT}/runs/f5"


def _resolve_xtts_ckpt_dir() -> str:
    """Newest ``runs/xtts/xtts_finetune_danil-*`` dir that has a config.json."""
    cands = sorted(
        d for d in glob.glob(f"{ROOT}/runs/xtts/xtts_finetune_danil-*")
        if os.path.isfile(os.path.join(d, "config.json"))
    )
    return cands[-1] if cands else f"{ROOT}/runs/xtts/MISSING_xtts_finetune_run"


XTTS_CKPT_DIR = _resolve_xtts_ckpt_dir()

EXPECTED_KERNEL_PY = f"{ROOT}/.venv/bin/python"


# --------------------------------------------------------------------------- #
# Voice descriptors                                                           #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Voice:
    key: str
    label: str
    env: str            # conda env name (also the key into ENV_PY / NPP)
    kind: str           # "engine_server" | "f5_server" | "xtts_server"
    module: str         # python -m <module>
    port: int
    engine: str | None = None        # engine_server --engine value
    checkpoint_dir: str | None = None  # fine-tuned servers /load_checkpoint
    ref_text: str = ""               # engine_server /synth ref_text (F5 needs it)
    langs: tuple = ("en",)           # languages this voice can speak


# German is cross-lingual (English reference -> German output). F5 fine-tuned is
# an English-only model so it stays ("en",); the other four reach German via the
# F5 German checkpoint / XTTS's multilingual base / ChatterboxMultilingualTTS.
EN_DE = ("en", "de")

VOICES: list[Voice] = [
    Voice("f5_finetuned", "Daniil — F5 (fine-tuned) ⭐", env="f5tts",
          kind="f5_server", module="voxclone.serve.f5_server", port=9892,
          checkpoint_dir=F5_CKPT_DIR, langs=("en",)),
    Voice("f5_zeroshot", "Daniil — F5 (zero-shot)", env="f5tts",
          kind="engine_server", module="voxclone.serve.engine_server", port=9882,
          engine="f5", ref_text=REF_TEXT, langs=EN_DE),
    Voice("xtts_finetuned", "Daniil — XTTS (fine-tuned)", env="xtts",
          kind="xtts_server", module="voxclone.serve.xtts_server", port=9891,
          checkpoint_dir=XTTS_CKPT_DIR, langs=EN_DE),
    Voice("xtts_zeroshot", "Daniil — XTTS (zero-shot)", env="xtts",
          kind="engine_server", module="voxclone.serve.engine_server", port=9881,
          engine="xtts", ref_text="", langs=EN_DE),
    Voice("chatterbox", "Daniil — Chatterbox", env="chatterbox",
          kind="engine_server", module="voxclone.serve.engine_server", port=9883,
          engine="chatterbox", ref_text="", langs=EN_DE),
]
VOICES_BY_KEY: dict[str, Voice] = {v.key: v for v in VOICES}
DEFAULT_VOICE = "f5_finetuned"
DEFAULT_GERMAN_VOICE = "f5_zeroshot"   # German-specialised F5 checkpoint


def voices_for(language: str) -> list[Voice]:
    """The voices that can speak ``language`` (e.g. 'en' or 'de')."""
    return [v for v in VOICES if language in v.langs]


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no GPU / no IO)                                   #
# --------------------------------------------------------------------------- #

def worker_argv(voice: Voice) -> list[str]:
    """The exact argv to launch ``voice``'s server via its env's interpreter."""
    argv = [ENV_PY[voice.env], "-m", voice.module, "--port", str(voice.port)]
    if voice.kind == "engine_server":
        argv += ["--engine", voice.engine]
    return argv


def worker_env(voice: Voice, base_env: dict | None = None) -> dict:
    """Env dict for the worker subprocess — no conda activate needed.

    The notebook kernel leaks env vars that break a different-env worker: the
    Jupyter kernel sets ``MPLBACKEND`` to its inline backend (absent in the
    engine envs, so matplotlib's import crashes), and the ``.venv`` sets
    ``VIRTUAL_ENV`` / sometimes ``PYTHONHOME`` / ``CONDA_PREFIX``. We pin a safe
    matplotlib backend and drop the venv/conda leak vars.
    """
    ev = dict(os.environ if base_env is None else base_env)
    ev["PYTHONPATH"] = f"{ROOT}/src"
    ld = ev.get("LD_LIBRARY_PATH", "")
    ev["LD_LIBRARY_PATH"] = NPP[voice.env] + (":" + ld if ld else "")
    ev["HF_HUB_OFFLINE"] = "1"        # caches are warm -> pure local reads
    ev["TRANSFORMERS_OFFLINE"] = "1"
    ev["MPLBACKEND"] = "Agg"          # kernel's inline backend isn't in the worker env
    for leak in ("CONDA_PREFIX", "VIRTUAL_ENV", "PYTHONHOME"):
        ev.pop(leak, None)            # don't leak the kernel/.venv interpreter context
    # NB: user-site (~/.local) is intentionally left ENABLED — torch in these envs
    # resolves typing_extensions from ~/.local, so PYTHONNOUSERSITE=1 would break it.
    if voice.env == "xtts":
        ev["COQUI_TOS_AGREED"] = "1"  # avoids the Coqui license prompt
    return ev


def synth_request(voice: Voice, text: str, language: str = "en",
                  ref_path: str | None = None,
                  ref_text: str | None = None) -> tuple[str, dict]:
    """(url_suffix, json body) for the per-utterance synthesis call.

    ``language`` rides through ``params`` for the engine_server voices (the
    adapters lazily load the German checkpoint / multilingual model on 'de') and
    through the ``language`` field for the XTTS fine-tuned server.

    ``ref_path`` / ``ref_text`` override the cloned speaker. By default every voice
    clones Daniil (``REF_CLIP`` + its transcript ``REF_TEXT``); passing a different
    clip makes a **zero-shot** engine_server voice clone *that* speaker instead —
    this is what lets the notebook clone the listener's own voice. (F5 conditions
    on ``ref_text``, so a custom clip should carry its own transcript; XTTS/
    Chatterbox ignore it. The fine-tuned servers carry Daniil in their weights, so
    a custom reference there is meaningless — ``VoiceStudio.say`` blocks it.)
    """
    ref = ref_path or REF_CLIP
    if voice.kind == "engine_server":
        rt = voice.ref_text if ref_text is None else ref_text
        return "/synth", {"text": text, "ref_path": ref,
                          "ref_text": rt, "params": {"language": language}}
    if voice.kind == "f5_server":  # English-only Daniil fine-tune
        return "/tts", {"text": text, "reference_clip": ref,
                        "prompt_text": REF_TEXT if ref_text is None else ref_text}
    if voice.kind == "xtts_server":
        return "/tts", {"text": text, "reference_clip": ref, "language": language}
    raise ValueError(f"unknown kind {voice.kind!r}")


def load_request(voice: Voice) -> tuple[str, dict] | None:
    """(url_suffix, body) for the one-time checkpoint load, or None (zero-shot)."""
    if voice.kind in ("f5_server", "xtts_server"):
        return "/load_checkpoint", {"checkpoint_dir": voice.checkpoint_dir}
    return None


def readiness_kind(voice: Voice) -> str:
    """All studio servers expose GET /health, so readiness is identity-aware
    (a foreign process that merely holds the port does not satisfy it)."""
    return "http_health"


# --------------------------------------------------------------------------- #
# Low-level IO helpers                                                         #
# --------------------------------------------------------------------------- #

def _port_up(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_health(port: int, host: str = "127.0.0.1", timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _post_bytes(url: str, body: dict, timeout: float) -> bytes:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _pid_is_our_worker(pid: int, port: int) -> bool:
    """True only if /proc/<pid> is one of our engine workers bound to ``port``.

    Guards the pidfile-driven kill: a stale pidfile left by a crashed kernel can
    name a PID that the OS has since recycled for an unrelated (possibly
    user-owned) process. We confirm identity from the live cmdline before
    signalling, so we never kill something that just happens to reuse the PID.
    """
    try:
        blob = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "voxclone.serve." in blob and f" {port}" in blob


def _kill_pg(pid: int, grace: float = 8.0) -> None:
    """SIGTERM the process group, then SIGKILL after a grace period."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _pidfile(port: int) -> Path:
    return Path(f"/tmp/vox_studio_{port}.pid")


def _logfile(voice: Voice) -> Path:
    return Path(f"/tmp/vox_{voice.key}_{voice.port}.log")


# --------------------------------------------------------------------------- #
# Pre-flight checks (used by the notebook's first cell)                        #
# --------------------------------------------------------------------------- #

ASSET_PATHS = [
    REF_CLIP,
    f"{ROOT}/runs/f5/model_last.pt",
    f"{ROOT}/runs/f5/vocab.txt",
    f"{XTTS_CKPT_DIR}/config.json",
    f"{ROOT}/third_party/xtts_v2_model/config.json",
    f"{ROOT}/third_party/xtts_v2_model/model.pth",
    # The XTTS fine-tune's config.json points its tokenizer at the base vocab.json,
    # so both XTTS voices need these base files present too.
    f"{ROOT}/third_party/xtts_v2_model/vocab.json",
    f"{ROOT}/third_party/xtts_v2_model/dvae.pth",
    f"{ROOT}/third_party/xtts_v2_model/mel_stats.pth",
]


def missing_assets() -> list[str]:
    return [p for p in ASSET_PATHS if not os.path.exists(p)]


def missing_envs() -> list[str]:
    return [e for e, py in ENV_PY.items() if not os.path.exists(py)]


# --------------------------------------------------------------------------- #
# The studio                                                                  #
# --------------------------------------------------------------------------- #

class VoiceStudio:
    """Keeps one warm engine worker at a time and clones Daniil's voice."""

    def __init__(self, outdir: str | None = None, on_status=None):
        self.outdir = Path(outdir or f"{ROOT}/notebooks/clone_outputs")
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._status_cb = on_status or (lambda m: print(m, flush=True))
        self._proc: subprocess.Popen | None = None
        self._voice: str | None = None
        self._loaded = False
        self._counter = 0
        atexit.register(self.shutdown)

    # -- public API -------------------------------------------------------- #

    def say(self, voice_key: str, text: str, language: str = "en", timeout: float = 300.0,
            ref_path: str | None = None, ref_text: str | None = None):
        """Synthesize ``text`` in ``voice_key`` / ``language``; returns (wav, sr, Path).

        ``ref_path`` (+ optional ``ref_text`` transcript) clones a **custom** speaker
        instead of Daniil — e.g. the listener's own ~6-12 s clip. Only the zero-shot
        voices (``f5_zeroshot`` / ``xtts_zeroshot`` / ``chatterbox``) can do this; the
        fine-tuned voices are Daniil baked into the weights and reject a custom clip.
        """
        if not text or not text.strip():
            raise ValueError("text is empty")
        voice = VOICES_BY_KEY[voice_key]
        if language not in voice.langs:
            ok = ", ".join(v.label for v in voices_for(language)) or "(none)"
            raise ValueError(
                f"{voice.label} is English-only — it can't speak '{language}'. "
                f"For that language pick: {ok}."
            )
        if ref_path is not None and voice.kind != "engine_server":
            zs = ", ".join(v.label for v in VOICES if v.kind == "engine_server")
            raise ValueError(
                f"{voice.label} is a fine-tuned model of Daniil's voice — it can't clone "
                f"a custom reference. To clone your own voice pick a zero-shot voice: {zs}."
            )
        self._ensure_worker(voice, timeout=timeout)
        suffix, body = synth_request(voice, text, language, ref_path=ref_path, ref_text=ref_text)
        url = f"http://127.0.0.1:{voice.port}{suffix}"
        self._status(f"… generating with {voice.label}")
        try:
            wav_bytes = _post_bytes(url, body, timeout=timeout)
        except urllib.error.HTTPError as e:
            # The worker answered with an error (bad input / OOM / broken model).
            # Surface it — do NOT tear a healthy worker down and retry blindly.
            try:
                body_txt = e.read().decode("utf-8", "replace")[:1000]
            except Exception:
                body_txt = ""
            raise RuntimeError(
                f"{voice.label} server error (HTTP {e.code}). {body_txt}".strip()
            ) from e
        except (urllib.error.URLError, ConnectionError, socket.timeout):
            # Connection-level failure: the worker may have died mid-session —
            # relaunch once and retry.
            self._status(f"→ {voice.label} worker unresponsive; restarting once…")
            self._teardown()
            self._ensure_worker(voice, timeout=timeout)
            wav_bytes = _post_bytes(url, body, timeout=timeout)
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype("float32")
        path = self._save(voice, audio, sr)
        self._status(f"✓ {voice.label}: saved {path.name}")
        return audio.astype("float32"), int(sr), path

    def restart(self, voice_key: str | None = None) -> None:
        """Tear down the current worker so the next say() relaunches it."""
        self._teardown()
        self._status("↺ engine stopped — it will reload on the next generate.")

    def shutdown(self) -> None:
        try:
            self._teardown()
        except Exception:
            pass

    # -- worker lifecycle -------------------------------------------------- #

    def _ensure_worker(self, voice: Voice, timeout: float) -> None:
        ready = (
            self._voice == voice.key
            and self._alive()
            and _port_up(voice.port)
            and (self._loaded or voice.kind == "engine_server")
        )
        if ready:
            return
        self._teardown()
        self._launch(voice)
        self._wait_ready(voice, timeout)
        self._voice = voice.key
        load = load_request(voice)
        if load is not None:
            suffix, body = load
            self._status(
                f"… loading {voice.label} onto the GPU (first run, ~30–90s)…"
            )
            try:
                _post_bytes(f"http://127.0.0.1:{voice.port}{suffix}", body,
                            timeout=max(timeout, 600.0))
            except Exception:
                # A failed load leaves a worker holding VRAM — free it now
                # instead of relying on the next call's reuse guard.
                self._teardown()
                raise
            self._loaded = True

    def _launch(self, voice: Voice) -> None:
        self._reclaim_port(voice.port)
        log = _logfile(voice)
        self._status(
            f"… starting {voice.label} in conda env '{voice.env}' "
            f"(background, no terminal)…"
        )
        self._proc = subprocess.Popen(
            worker_argv(voice), cwd=ROOT, env=worker_env(voice),
            stdout=open(log, "wb"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _pidfile(voice.port).write_text(str(self._proc.pid))
        self._loaded = False

    def _wait_ready(self, voice: Voice, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                _pidfile(voice.port).unlink(missing_ok=True)
                raise RuntimeError(
                    f"{voice.label} worker exited during startup.\n{self._log_tail(voice)}"
                )
            if _http_health(voice.port):  # identity-aware: only OUR server answers /health
                return
            time.sleep(1.0)
        raise TimeoutError(
            f"{voice.label} did not become ready within {timeout:.0f}s.\n{self._log_tail(voice)}"
        )

    def _reclaim_port(self, port: int) -> None:
        """Kill a leftover worker from a previous run (idempotent re-run).

        Only signals a pidfile PID we can positively identify as our own worker
        (PID reuse after a kernel crash means a stale PID may now be someone
        else's process). If the port stays held by a foreign process, fail loud
        rather than launch into a silent bind failure.
        """
        pf = _pidfile(port)
        if pf.exists():
            try:
                pid = int(pf.read_text().strip())
            except (ValueError, OSError):
                pid = None
            if pid and _pid_is_our_worker(pid, port):
                _kill_pg(pid)
            pf.unlink(missing_ok=True)
        deadline = time.monotonic() + 10.0
        while _port_up(port) and time.monotonic() < deadline:
            time.sleep(0.3)
        if _port_up(port):
            raise RuntimeError(
                f"Port {port} is still in use by another process — cannot start the "
                f"engine. Free that port (or restart the kernel) and try again."
            )

    def _teardown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            _kill_pg(self._proc.pid)
        if self._voice is not None:
            _pidfile(VOICES_BY_KEY[self._voice].port).unlink(missing_ok=True)
        self._proc = None
        self._voice = None
        self._loaded = False

    # -- misc -------------------------------------------------------------- #

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _save(self, voice: Voice, audio: np.ndarray, sr: int) -> Path:
        self._counter += 1
        path = self.outdir / f"{voice.key}_{self._counter:03d}.wav"
        sf.write(str(path), np.asarray(audio, dtype="float32"), int(sr))
        return path

    def _log_tail(self, voice: Voice, n: int = 25) -> str:
        log = _logfile(voice)
        try:
            lines = log.read_text(errors="replace").splitlines()
        except OSError:
            return "(no log captured)"
        return "  " + "\n  ".join(lines[-n:])

    def _status(self, msg: str) -> None:
        self._status_cb(msg)


# --------------------------------------------------------------------------- #
# Module-level convenience (one shared studio for the notebook)               #
# --------------------------------------------------------------------------- #

_STUDIO: VoiceStudio | None = None


def get_studio(**kwargs) -> VoiceStudio:
    global _STUDIO
    if _STUDIO is None:
        _STUDIO = VoiceStudio(**kwargs)
    return _STUDIO


def say(voice_key: str, text: str, language: str = "en", **kwargs):
    return get_studio().say(voice_key, text, language=language, **kwargs)


def restart(voice_key: str | None = None) -> None:
    get_studio().restart(voice_key)
