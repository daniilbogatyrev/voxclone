"""Zero-shot cross-engine eval on the existing danil clones (003c/004/005/006).

Scores the four pre-rendered zero-shot clips with the voxclone metric functions
(this is NOT run_eval, which regenerates from text via synth adapters that don't exist
yet for XTTS/F5/Chatterbox — those are Plan 07). It still exercises the real model
wiring (ECAPA / WhisperX / UTMOS) on real audio for the first time.

  - similarity : ECAPA-TDNN speaker-embedding cosine to a held-out real-danil centroid,
                 expressed as a fraction of the real-vs-real ceiling (picker.combined_score)
  - naturalness: UTMOS MOS (SpeechMOS utmos22_strong via torch.hub)
  - wer        : WhisperX large-v3 transcription vs the shared text, Whisper-normalized

CAVEATS (read before trusting ABSOLUTE numbers — the cross-engine RANKING is still fair):
  * Only ~14 s of real danil audio exists (danill.wav); ref9 (9 s of it) conditioned ALL
    four generations. The ceiling is built from two ~7 s halves of that single take -> a
    "same-session" ceiling, and similarity is reference-biased. BUT every engine used the
    SAME ref9, so the bias is equal across them -> the ranking is fair, the absolute
    "% of ceiling" is optimistic.
  * Naturalness uses UTMOS22 (SpeechMOS), not the sarulab UTMOSv2 named in naturalness.py
    (uninstalled / API unverified). Fine for ranking; the absolute MOS scale differs slightly.
  * ZERO-SHOT baseline only. The project decision is to fine-tune XTTS-v2 + F5-TTS and pick
    the winner by this same metric on the fine-tuned outputs.
"""
from pathlib import Path
import json
import numpy as np
import torch

from voxclone.common.audio import load_audio
from voxclone.common.config import EvalConfig
from voxclone.eval.normalize import normalize_for_eval
from voxclone.eval.similarity import load_ecapa, similarity_score, reference_ceiling
from voxclone.eval.wer import wer
from voxclone.eval.picker import combined_score, pick_best
from voxclone.eval.report import write_report
from voxclone.prep.transcribe import load_transcriber, transcribe_clip

ROOT = Path("/home/prada/code/danill")
EXP = ROOT / "experiments/danil/experiment"
REF = ROOT / "experiments/danil/reference"

SHARED = ("Hello — this is a quick zero-shot test of the cloned voice, reading a couple of "
          "natural English sentences aloud.")

# All four on the SAME sentence now (003c regenerated to match 004/005/006).
CLIPS = {
    "gptsovits_v2pro": EXP / "003c_v2pro_zeroshot_shared.wav",
    "xtts_v2":         EXP / "004_xtts_zeroshot.wav",
    "f5_tts":          EXP / "005_f5_zeroshot.wav",
    "chatterbox":      EXP / "006_chatterbox_zeroshot.wav",
}
HELD_OUT_REAL = [REF / "heldout_a.wav", REF / "heldout_b.wav"]


def load_utmos_speechmos():
    """SpeechMOS utmos22_strong — input is 16 kHz mono float (from normalize_for_eval)."""
    model = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)

    def score(audio_16k: np.ndarray, sr: int) -> float:
        wav = torch.from_numpy(np.ascontiguousarray(audio_16k)).unsqueeze(0)
        with torch.no_grad():
            return float(model(wav, sr))

    return score


def embed_clip(path, embedder):
    audio, sr = load_audio(path)
    norm, nsr = normalize_for_eval(audio, sr)  # 16k / mono / -23 LUFS / peak-limited
    return np.asarray(embedder(norm)), norm, nsr


def main():
    cfg = EvalConfig()
    weights = cfg.weights()
    embedder = load_ecapa()
    transcriber = load_transcriber("large-v3", "cuda")
    utmos = load_utmos_speechmos()

    # Real-vs-real ceiling from the held-out danil halves (same front end as gen clips).
    real_emb = np.stack([embed_clip(p, embedder)[0] for p in HELD_OUT_REAL])
    ceiling = reference_ceiling(real_emb)

    metrics, rows = {}, []
    for name, path in CLIPS.items():
        emb, norm, nsr = embed_clip(path, embedder)
        sim = similarity_score(np.stack([emb]), real_emb)
        hyp = transcribe_clip(norm, nsr, transcriber).text
        w = wer(SHARED, hyp)
        nat = utmos(norm, nsr)
        score = combined_score(sim, nat, w, weights, ceiling=ceiling)
        metrics[name] = {"similarity": sim, "naturalness": nat, "wer": w,
                         "score": score, "hyp": hyp}
        rows.append({"checkpoint": name, "similarity": sim, "naturalness": nat,
                     "wer": w, "score": score})

    best = pick_best(metrics, weights, wer_dq_threshold=cfg.wer_dq_threshold, ceiling=ceiling)
    write_report(ROOT / "reports/eval_zeroshot.md", rows, ceiling)

    print(f"\nceiling (real-vs-real, same-session): {ceiling:.4f}   "
          f"(85% target = {0.85 * ceiling:.4f})")
    print(f"weights: sim {weights['similarity']} / nat {weights['naturalness']} / "
          f"wer {weights['wer']}   | wer_dq = {cfg.wer_dq_threshold}\n")
    hdr = f"{'engine':<17}{'sim':>8}{'sim/ceil':>10}{'UTMOS':>8}{'WER':>8}{'score':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in sorted(metrics.items(), key=lambda kv: -kv[1]["score"]):
        sc = min(m["similarity"] / ceiling, 1.0) if ceiling > 0 else m["similarity"]
        print(f"{name:<17}{m['similarity']:>8.4f}{sc:>10.4f}{m['naturalness']:>8.3f}"
              f"{m['wer']:>8.4f}{m['score']:>8.4f}")
    print(f"\nbest (naturalness-weighted, WER-DQ > {cfg.wer_dq_threshold}): {best}\n")
    print("transcripts (WhisperX large-v3):")
    for name, m in metrics.items():
        print(f"  {name:<17} {m['hyp']!r}")

    out = {
        "date": "2026-05-27", "kind": "zeroshot_cross_engine_eval", "shared_text": SHARED,
        "ceiling": ceiling, "weights": weights, "wer_dq_threshold": cfg.wer_dq_threshold,
        "held_out": [p.name for p in HELD_OUT_REAL],
        "naturalness_model": "speechmos:utmos22_strong", "asr_model": "whisperx:large-v3",
        "results": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk, vv in v.items()} for k, v in metrics.items()},
        "best": best,
    }
    (ROOT / "experiments/eval_zeroshot_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/eval_zeroshot.md + experiments/eval_zeroshot_results.json")


if __name__ == "__main__":
    main()
