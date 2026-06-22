import numpy as np
from voxclone.common.audio import load_audio
from voxclone.eval.normalize import normalize_for_eval
from voxclone.eval.similarity import similarity_score, reference_ceiling
from voxclone.eval.wer import wer
from voxclone.eval.naturalness import utmos_score, aggregate_utmos
from voxclone.eval.picker import combined_score, pick_best
from voxclone.eval.report import write_report
from voxclone.prep.transcribe import transcribe_clip


class _CallableTranscriberAdapter:
    """Wraps a plain callable ``fn(audio, sr) -> list[dict]`` so it satisfies the
    ``model.transcribe(audio, sr)`` protocol expected by ``transcribe_clip``."""

    def __init__(self, fn):
        self._fn = fn

    def transcribe(self, audio: np.ndarray, sr: int) -> list[dict]:
        return self._fn(audio, sr)


def run_eval(checkpoints: dict, held_out: list[tuple[str, str]], reference_clip: str,
             embedder, transcriber, utmos_model, weights: dict,
             report_path="reports/eval.md", wer_dq_threshold: float | None = None,
             registry=None, ref_text: str = "") -> dict:
    if not hasattr(transcriber, "transcribe"):
        transcriber = _CallableTranscriberAdapter(transcriber)
    if not checkpoints or not held_out:
        raise ValueError("checkpoints and held_out must be non-empty")

    # Real held-out clips go through the SAME front end as generated clips, so the
    # similarity ceiling is on the same footing (16 kHz / mono / -23 LUFS).
    real_embs = []
    for _, path in held_out:
        audio, sr = load_audio(path)
        norm, nsr = normalize_for_eval(audio, sr)
        real_embs.append(embedder(norm))
    real_emb = np.stack(real_embs)
    ceiling = reference_ceiling(real_emb)

    metrics: dict[str, dict] = {}
    rows = []
    for name, synth in checkpoints.items():
        gen_embs, wers, utmos_scores = [], [], []
        for text, _ in held_out:
            # F5 conditions on the reference clip's transcript (params['prompt_text']);
            # engines that ignore it are unaffected. Default '' preserves prior behavior.
            audio, sr = synth.synthesize(text, reference_clip, {"prompt_text": ref_text})
            norm, nsr = normalize_for_eval(audio, sr)
            gen_embs.append(embedder(norm))
            hyp = transcribe_clip(norm, nsr, transcriber).text
            wers.append(wer(text, hyp))
            utmos_scores.append(utmos_score(norm, nsr, utmos_model))
        sim = similarity_score(np.stack(gen_embs), real_emb)
        wer_mean = float(np.mean(wers))
        utmos_mean = aggregate_utmos(utmos_scores)
        score = combined_score(sim, utmos_mean, wer_mean, weights, ceiling=ceiling)
        metrics[name] = {"similarity": sim, "naturalness": utmos_mean,
                         "wer": wer_mean, "score": score}
        rows.append({"checkpoint": name, **metrics[name]})
        # Persist this scored candidate so serve can resolve best_checkpoint. The candidate
        # name is the registry model key (engine[:label]); the synth's bound checkpoint is the
        # path; metrics already carries the 'score' that best_checkpoint maximizes over.
        if registry is not None:
            registry.register(name, getattr(synth, "checkpoint", name), metrics[name])
    best = pick_best(metrics, weights, wer_dq_threshold=wer_dq_threshold, ceiling=ceiling)
    write_report(report_path, rows, ceiling)
    return {"metrics": metrics, "best": best, "ceiling": ceiling}
