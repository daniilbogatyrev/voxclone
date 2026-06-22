from itertools import combinations
import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def reference_ceiling(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine across real clips — the practical 'same speaker' anchor."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    pairs = list(combinations(range(len(embeddings)), 2))
    if not pairs:
        return 1.0
    return float(np.mean([cosine(embeddings[i], embeddings[j]) for i, j in pairs]))

def similarity_score(gen: np.ndarray, real: np.ndarray) -> float:
    """Mean cosine of each generated embedding to the real-clip centroid."""
    gen = np.asarray(gen, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    centroid = real.mean(axis=0)
    return float(np.mean([cosine(g, centroid) for g in gen]))

def load_ecapa():  # pragma: no cover (model)
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    def embed(audio_16k_mono: np.ndarray) -> np.ndarray:
        # input is already 16 kHz mono (ECAPA's required rate), from normalize_for_eval
        sig = torch.tensor(np.asarray(audio_16k_mono, dtype=np.float32)).unsqueeze(0)
        return model.encode_batch(sig).squeeze().detach().cpu().numpy()
    return embed
