import io
from pathlib import Path
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

class SynthRequest(BaseModel):
    text: str
    # Default is a SHORT engine key (the train->registry->serve naming contract); f5 is
    # the project's confirmed #1 winner. create_app stays engine-agnostic -- this string
    # is the only engine-specific knob here; provider() resolves the actual *Synth.
    model: str = "f5"
    seed: int | None = None
    # Transcript of the reference clip, threaded to the synth as params['prompt_text'].
    # F5 conditions on it; engines that ignore prompt_text are unaffected. None falls
    # back to create_app's configured ref_text default.
    ref_text: str | None = None

    @field_validator("text")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must be non-empty")
        return v

def create_app(synth_provider, reference_clip: str, ref_text: str = "",
               frontend_dir: str = "frontend") -> FastAPI:
    app = FastAPI(title="voxclone")

    fdir = Path(frontend_dir)
    if fdir.exists():
        app.mount("/static", StaticFiles(directory=str(fdir)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(fdir / "index.html"))

    @app.post("/synthesize")
    def synthesize(req: SynthRequest) -> Response:
        try:
            synth = synth_provider(req.model)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown model {req.model}")
        # F5 needs the reference clip's transcript as params['prompt_text']; a per-request
        # ref_text overrides the app-level default. Engines that ignore it are unaffected.
        prompt_text = req.ref_text if req.ref_text is not None else ref_text
        params = {"seed": req.seed, "prompt_text": prompt_text}
        audio, sr = synth.synthesize(req.text, reference_clip, params)
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    return app


# --------------------------------------------------------------------------- #
# Studio-backed live demo (same backend as the notebook: 5 voices, EN/DE)      #
# --------------------------------------------------------------------------- #

class CloneRequest(BaseModel):
    """Live-demo request: which studio voice, what text, what language."""
    voice: str
    text: str
    language: str = "en"

    @field_validator("text")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must be non-empty")
        return v


class RestartRequest(BaseModel):
    voice: str | None = None


def create_studio_app(studio=None, frontend_dir: str = "frontend") -> FastAPI:
    """Web UI backed by the SAME ``voxclone.clone.studio`` the notebook uses: 5 voices,
    English/German, real generation (the studio auto-launches each engine's conda-env
    worker). ``studio`` is injectable for tests; it defaults to the real module, imported
    lazily on first request so importing this module stays light/GPU-free.

    Single-user demo: the studio keeps ONE warm worker at a time, so concurrent requests
    are not supported (the front end disables Generate while a clip is in flight, and
    switching voice tears the previous worker down to free VRAM).
    """
    app = FastAPI(title="voxclone-studio")
    # The presentation deck (presentation/voxclone.html) is opened from a different
    # origin -- file:// or its own http.server port -- yet calls /voices, /languages
    # and /clone here. Allow any origin: this is a local, single-user demo backend.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state: dict = {"studio": studio}

    def _studio():
        if state["studio"] is None:
            from voxclone.clone import studio as _s  # pragma: no cover (real module)
            state["studio"] = _s
        return state["studio"]

    fdir = Path(frontend_dir)
    if fdir.exists():
        app.mount("/static", StaticFiles(directory=str(fdir)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(fdir / "index.html"))

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/languages")
    def languages() -> dict:
        return {"languages": [{"code": "en", "label": "English"},
                              {"code": "de", "label": "Deutsch (German)"}]}

    @app.get("/voices")
    def voices(language: str = "en") -> dict:
        s = _studio()
        vs = [{"key": v.key, "label": v.label} for v in s.voices_for(language)]
        default = s.DEFAULT_GERMAN_VOICE if language == "de" else s.DEFAULT_VOICE
        return {"language": language, "voices": vs, "default": default}

    @app.post("/clone")
    def clone(req: CloneRequest) -> Response:
        s = _studio()
        try:
            wav, sr, _ = s.say(req.voice, req.text, language=req.language)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown voice {req.voice!r}")
        except ValueError as e:                # empty text / voice can't speak language
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:                 # worker OOM / timeout / load failure
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        buf = io.BytesIO()
        sf.write(buf, np.asarray(wav, dtype=np.float32), int(sr), format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    @app.post("/restart")
    def restart(req: RestartRequest) -> dict:
        _studio().restart(req.voice)
        return {"restarted": req.voice}

    return app
