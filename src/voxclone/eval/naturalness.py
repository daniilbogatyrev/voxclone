import numpy as np

def utmos_score(audio: np.ndarray, sr: int, model) -> float:
    return float(model(audio, sr))

def aggregate_utmos(scores: list[float]) -> float:
    return float(np.mean(scores)) if scores else 0.0

def load_utmos():  # pragma: no cover (model)
    import utmosv2
    predictor = utmosv2.create_model(pretrained=True)
    def score(audio: np.ndarray, sr: int) -> float:
        return float(predictor.predict(audio=audio, sr=sr))
    return score


def _torch_hub():  # pragma: no cover (indirection seam so tests can stub torch.hub)
    import torch
    return torch.hub


def _as_tensor(audio_16k: np.ndarray):  # pragma: no cover (real torch path)
    import torch
    return torch.from_numpy(np.ascontiguousarray(audio_16k)).unsqueeze(0)


def load_speechmos_utmos():
    """UTMOS via SpeechMOS utmos22_strong (torch.hub). Stand-in for the uninstalled sarulab
    UTMOSv2 that the (unused) load_utmos targets. Input is 16 kHz mono float (from
    normalize_for_eval).
    NOTE: torch.hub.load downloads + runs an external repo (one-time user authorization)."""
    hub = _torch_hub()
    model = hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
    # Real torch.hub exposes torch.no_grad via its module; the test stub (SimpleNamespace) does
    # not, so the heavy tensor/no_grad path is taken only against the real model.
    real = hasattr(hub, "load") and getattr(hub, "__name__", "") == "torch.hub"

    def score(audio_16k: np.ndarray, sr: int) -> float:
        if real:  # pragma: no cover (real torch path; tests pass a stub model)
            import torch
            with torch.no_grad():
                return float(model(_as_tensor(audio_16k), sr))
        return float(model(np.ascontiguousarray(audio_16k), sr))

    return score
