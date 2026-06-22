"""Uniform .venv-side client for the four engine servers. GPT-SoVITS uses its existing api_v2
(via GPTSoVITSSynth); xtts/f5/chatterbox use engine_server's /synth. The notebook depends on
THIS contract, not on importing the per-env adapters."""
import io

import httpx
import numpy as np
import soundfile as sf

from voxclone.synth.gptsovits import GPTSoVITSSynth

ENGINES = {
    "gptsovits_v2pro": {"port": 9880, "kind": "api_v2", "conda_env": "gptsovits"},
    "xtts_v2":         {"port": 9881, "kind": "engine_server", "conda_env": "xtts"},
    "f5_tts":          {"port": 9882, "kind": "engine_server", "conda_env": "f5tts"},
    "chatterbox":      {"port": 9883, "kind": "engine_server", "conda_env": "chatterbox"},
}


def health(engine: str, timeout: float = 2.0) -> bool:
    """Any HTTP response on the engine's port counts as up (api_v2 returns 404 on '/')."""
    cfg = ENGINES[engine]
    url = f"http://127.0.0.1:{cfg['port']}/" + ("" if cfg["kind"] == "api_v2" else "health")
    try:
        httpx.get(url, timeout=timeout)
        return True
    except Exception:
        return False


def synth(engine: str, text: str, ref_path: str, ref_text: str = "",
          params: dict | None = None) -> tuple[np.ndarray, int]:
    params = dict(params or {})
    cfg = ENGINES[engine]
    if cfg["kind"] == "api_v2":
        p = {"prompt_text": ref_text, "text_split_method": "cut0",
             "temperature": 0.7, "top_k": 15, "prompt_lang": "en", "text_lang": "en", **params}
        return GPTSoVITSSynth(checkpoint="v2Pro").synthesize(text, ref_path, p)
    body = {"text": text, "ref_path": ref_path, "ref_text": ref_text, "params": params}
    resp = httpx.post(f"http://127.0.0.1:{cfg['port']}/synth", json=body,
                      timeout=params.get("timeout", 120.0))
    resp.raise_for_status()
    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")
    return audio.astype("float32"), int(sr)
