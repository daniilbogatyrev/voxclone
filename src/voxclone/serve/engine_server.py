"""Parametrized FastAPI server for one TTS engine (run one per conda env):

  PYTHONPATH=src conda run -n xtts \
      python -m voxclone.serve.engine_server --engine xtts --port 9881

Exposes POST /synth {text, ref_path, ref_text, params} -> WAV bytes (float32) and GET /health.
The heavy engine lib is imported lazily inside the adapter's load()."""
import argparse
import io

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel


def _default_adapter(engine: str):  # pragma: no cover (real dispatch needs engine libs)
    if engine == "xtts":
        from voxclone.synth.xtts import XTTSAdapter
        return XTTSAdapter()
    if engine == "f5":
        from voxclone.synth.f5 import F5Adapter
        return F5Adapter()
    if engine == "chatterbox":
        from voxclone.synth.chatterbox import ChatterboxAdapter
        return ChatterboxAdapter()
    raise ValueError(f"unknown engine {engine}")


class SynthReq(BaseModel):
    text: str
    ref_path: str
    ref_text: str = ""
    params: dict = {}


def make_app(engine: str, adapter_factory=None) -> FastAPI:
    app = FastAPI(title=f"voxclone-{engine}")
    adapter = (adapter_factory or (lambda: _default_adapter(engine)))()
    adapter.load()

    @app.get("/health")
    def health():
        return {"engine": engine, "ready": True}

    @app.post("/synth")
    def synth(req: SynthReq):
        audio, sr = adapter.synthesize(req.text, req.ref_path, req.ref_text, req.params)
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio, dtype="float32"), sr, format="WAV", subtype="FLOAT")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    return app


def main() -> None:  # pragma: no cover (launches a real server)
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["xtts", "f5", "chatterbox"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(make_app(args.engine), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
