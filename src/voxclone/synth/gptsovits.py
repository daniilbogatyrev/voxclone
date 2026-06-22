import io
import numpy as np
import soundfile as sf
import httpx

DEFAULT_SERVER_URL = "http://127.0.0.1:9880"

def _real_generate(text, reference_clip, checkpoint, params):
    """Synthesize via a running GPT-SoVITS api_v2 server. Returns (float32 mono audio, sr).

    Zero-shot uses the server's currently loaded (base) weights. To use a fine-tuned
    checkpoint, pass params['gpt_weights'] and params['sovits_weights'] (paths on the
    server's filesystem). `checkpoint` is retained for interface compatibility.
    Note: GPT-SoVITS v2Pro requires a non-empty params['prompt_text'] (the reference
    clip's transcript).
    """
    base = params.get("server_url", DEFAULT_SERVER_URL)
    timeout = params.get("timeout", 120.0)
    gpt_w = params.get("gpt_weights")
    sovits_w = params.get("sovits_weights")
    if gpt_w:
        httpx.get(f"{base}/set_gpt_weights", params={"weights_path": gpt_w},
                  timeout=timeout).raise_for_status()
    if sovits_w:
        httpx.get(f"{base}/set_sovits_weights", params={"weights_path": sovits_w},
                  timeout=timeout).raise_for_status()
    payload = {
        "text": text,
        "text_lang": params.get("text_lang", "en"),
        "ref_audio_path": reference_clip,
        "prompt_text": params.get("prompt_text", ""),
        "prompt_lang": params.get("prompt_lang", "en"),
        # 'cut4' is the English-safe default: 'cut5' splits on every comma and
        # collapses English text to silence. Callers may override via params.
        "text_split_method": params.get("text_split_method", "cut4"),
        "batch_size": params.get("batch_size", 1),
        "media_type": "wav",
        "streaming_mode": False,
    }
    for k in ("top_k", "top_p", "temperature", "seed", "sample_steps", "speed_factor"):
        if k in params:
            payload[k] = params[k]
    resp = httpx.post(f"{base}/tts", json=payload, timeout=timeout)
    resp.raise_for_status()
    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio.astype(np.float32), int(sr)

class GPTSoVITSSynth:
    def __init__(self, checkpoint: str, generate_fn=_real_generate):
        self.checkpoint = checkpoint
        self.generate_fn = generate_fn

    def synthesize(self, text: str, reference_clip: str,
                   params: dict) -> tuple[np.ndarray, int]:
        audio, sr = self.generate_fn(text, reference_clip, self.checkpoint, params)
        return np.asarray(audio, dtype=np.float32), int(sr)
