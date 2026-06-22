"""Fine-tuned vs zero-shot bake-off: score all conditions on the shared A/B sentences.

Extends ab_eval.py to the 2 fine-tuned models (F5, XTTS) alongside the 4 zero-shot
baselines, with the SAME metric pipeline + real-danil centroid + ceiling, so they're
directly comparable. Same caveats as ab_eval.py (reference-biased similarity, same-session
ceiling, SpeechMOS UTMOS22 standing in for UTMOSv2). Writes experiments/ab_eval_finetune_results.json
+ reports/eval_finetune.md.
"""
from pathlib import Path
import json
import numpy as np
import torch

from voxclone.common.audio import load_audio
from voxclone.common.config import EvalConfig
from voxclone.eval.normalize import normalize_for_eval
from voxclone.eval.similarity import load_ecapa, reference_ceiling, cosine
from voxclone.eval.wer import wer
from voxclone.eval.picker import combined_score
from voxclone.prep.transcribe import load_transcriber, transcribe_clip

ROOT = Path("/home/prada/code/danill")
AB = ROOT / "experiments/danil/ab"
REF = ROOT / "experiments/danil/reference"
SENTS = json.loads((ROOT / "experiments/ab_sentences.json").read_text())
HELD_OUT_REAL = [REF / "heldout_a.wav", REF / "heldout_b.wav"]

# (dir, engine, mode)
CONDITIONS = [
    ("f5_finetuned", "F5-TTS", "fine-tuned"),
    ("xtts_finetuned", "XTTS-v2", "fine-tuned"),
    ("f5_tts", "F5-TTS", "zero-shot"),
    ("xtts_v2", "XTTS-v2", "zero-shot"),
    ("gptsovits_v2pro", "GPT-SoVITS", "zero-shot"),
    ("chatterbox", "Chatterbox", "zero-shot"),
]


def load_utmos_speechmos():
    model = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)

    def score(audio_16k, sr):
        wav = torch.from_numpy(np.ascontiguousarray(audio_16k)).unsqueeze(0)
        with torch.no_grad():
            return float(model(wav, sr))
    return score


def embed_clip(path, embedder):
    audio, sr = load_audio(path)
    norm, nsr = normalize_for_eval(audio, sr)
    return np.asarray(embedder(norm)), norm, nsr


def main():
    cfg = EvalConfig()
    weights = cfg.weights()
    embedder = load_ecapa()
    transcriber = load_transcriber("large-v3", "cuda")
    utmos = load_utmos_speechmos()

    real = [embed_clip(p, embedder) for p in HELD_OUT_REAL]
    real_emb = np.stack([e for e, _, _ in real])
    centroid = real_emb.mean(0)
    ceiling = reference_ceiling(real_emb)
    real_utmos = float(np.mean([utmos(n, sr) for _, n, sr in real]))

    rows = []
    for key, engine, mode in CONDITIONS:
        sims, wers, moss = [], [], []
        for i, text in enumerate(SENTS, 1):
            emb, norm, nsr = embed_clip(AB / key / f"s{i:02d}.wav", embedder)
            hyp = transcribe_clip(norm, nsr, transcriber).text
            sims.append(cosine(emb, centroid)); wers.append(wer(text, hyp)); moss.append(utmos(norm, nsr))
        sim, wer_m, ut = float(np.mean(sims)), float(np.mean(wers)), float(np.mean(moss))
        score = combined_score(sim, ut, wer_m, weights, ceiling=ceiling)
        rows.append({"key": key, "engine": engine, "mode": mode, "similarity": sim,
                     "pct_ceiling": min(sim / ceiling, 1.0) if ceiling > 0 else sim,
                     "naturalness": ut, "utmos_std": float(np.std(moss)),
                     "wer": wer_m, "score": score})

    rows.sort(key=lambda r: -r["score"])

    print(f"\nFINE-TUNED vs ZERO-SHOT bake-off | N={len(SENTS)} sentences/condition")
    print(f"ceiling (same-session) {ceiling:.4f} | real danil UTMOS {real_utmos:.3f} | "
          f"weights sim {weights['similarity']}/nat {weights['naturalness']}/wer {weights['wer']}\n")
    hdr = f"{'rank':<5}{'engine':<12}{'mode':<11}{'sim':>7}{'%ceil':>7}{'UTMOS':>8}{'WER':>8}{'score':>8}"
    print(hdr); print("-" * len(hdr))
    for r, m in enumerate(rows, 1):
        print(f"{r:<5}{m['engine']:<12}{m['mode']:<11}{m['similarity']:>7.3f}"
              f"{m['pct_ceiling'] * 100:>6.0f}%{m['naturalness']:>8.3f}{m['wer']:>8.3f}{m['score']:>8.3f}")

    out = {"date": "2026-06-01", "kind": "finetuned_vs_zeroshot_AB", "n_sentences": len(SENTS),
           "ceiling": ceiling, "real_utmos_heldout": real_utmos, "weights": weights,
           "wer_dq_threshold": cfg.wer_dq_threshold, "rows": rows}
    (ROOT / "experiments/ab_eval_finetune_results.json").write_text(json.dumps(out, indent=2))

    md = ["# Fine-tuned vs zero-shot bake-off", "",
          f"N={len(SENTS)} sentences/condition · real-vs-real ceiling **{ceiling:.4f}** · "
          f"real danil UTMOS {real_utmos:.3f} · weights sim {weights['similarity']}/nat "
          f"{weights['naturalness']}/wer {weights['wer']}.", "",
          "| rank | engine | mode | similarity | % ceiling | UTMOS | WER | score |",
          "|---|---|---|---|---|---|---|---|"]
    for r, m in enumerate(rows, 1):
        md.append(f"| {r} | {m['engine']} | {m['mode']} | {m['similarity']:.3f} | "
                  f"{m['pct_ceiling'] * 100:.0f}% | {m['naturalness']:.3f} | {m['wer']:.3f} | {m['score']:.3f} |")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/eval_finetune.md").write_text("\n".join(md) + "\n")
    print("\nwrote reports/eval_finetune.md + experiments/ab_eval_finetune_results.json")


if __name__ == "__main__":
    main()
