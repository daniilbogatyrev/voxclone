"""Robust zero-shot cross-engine A/B: N sentences per engine, metrics averaged.

Same methodology + caveats as eval_zeroshot.py (read its docstring): reference-biased
similarity, same-session ceiling, UTMOS22 (SpeechMOS) standing in for UTMOSv2, zero-shot
baseline only. Here we score experiments/danil/ab/<engine>/sNN.wav over the shared
sentence set (ab_sentences.json) and AVERAGE similarity / UTMOS / WER per engine, so the
ranking rests on N clips instead of one. Reuses the voxclone metric fns.
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
from voxclone.eval.picker import combined_score, pick_best
from voxclone.prep.transcribe import load_transcriber, transcribe_clip

ROOT = Path("/home/prada/code/danill")
AB = ROOT / "experiments/danil/ab"
REF = ROOT / "experiments/danil/reference"
SENTS = json.loads((ROOT / "experiments/ab_sentences.json").read_text())
ENGINES = ["gptsovits_v2pro", "xtts_v2", "f5_tts", "chatterbox"]
HELD_OUT_REAL = [REF / "heldout_a.wav", REF / "heldout_b.wav"]


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

    real_norm = [embed_clip(p, embedder) for p in HELD_OUT_REAL]
    real_emb = np.stack([e for e, _, _ in real_norm])
    centroid = real_emb.mean(0)
    ceiling = reference_ceiling(real_emb)
    real_utmos = float(np.mean([utmos(n, sr) for _, n, sr in real_norm]))

    summary, per_clip = {}, {}
    for eng in ENGINES:
        sims, wers, moss, clips = [], [], [], []
        for i, text in enumerate(SENTS, 1):
            emb, norm, nsr = embed_clip(AB / eng / f"s{i:02d}.wav", embedder)
            hyp = transcribe_clip(norm, nsr, transcriber).text
            w, m, s = wer(text, hyp), utmos(norm, nsr), cosine(emb, centroid)
            sims.append(s); wers.append(w); moss.append(m)
            clips.append({"i": i, "sim": round(s, 4), "wer": round(w, 4),
                          "utmos": round(m, 3), "hyp": hyp})
        sim, wer_mean, utmos_mean = float(np.mean(sims)), float(np.mean(wers)), float(np.mean(moss))
        score = combined_score(sim, utmos_mean, wer_mean, weights, ceiling=ceiling)
        summary[eng] = {"similarity": sim, "naturalness": utmos_mean, "wer": wer_mean,
                        "score": score, "sim_std": float(np.std(sims)),
                        "utmos_std": float(np.std(moss)), "wer_std": float(np.std(wers))}
        per_clip[eng] = clips

    best = pick_best(summary, weights, wer_dq_threshold=cfg.wer_dq_threshold, ceiling=ceiling)

    print(f"\nN={len(SENTS)} sentences/engine | ceiling (same-session) {ceiling:.4f} "
          f"(85% = {0.85 * ceiling:.4f}) | real danil UTMOS (held-out) {real_utmos:.3f}")
    print(f"weights sim {weights['similarity']} / nat {weights['naturalness']} / "
          f"wer {weights['wer']} | wer_dq {cfg.wer_dq_threshold}\n")
    hdr = (f"{'engine':<17}{'sim':>8}{'%ceil':>7}{'UTMOS':>8}{'±u':>6}"
           f"{'WER':>8}{'±w':>6}{'score':>8}")
    print(hdr)
    print("-" * len(hdr))
    for eng, m in sorted(summary.items(), key=lambda kv: -kv[1]["score"]):
        sc = min(m["similarity"] / ceiling, 1.0) if ceiling > 0 else m["similarity"]
        print(f"{eng:<17}{m['similarity']:>8.4f}{sc * 100:>6.0f}%{m['naturalness']:>8.3f}"
              f"{m['utmos_std']:>6.2f}{m['wer']:>8.4f}{m['wer_std']:>6.2f}{m['score']:>8.4f}")
    print(f"\nbest (naturalness-weighted, WER-DQ > {cfg.wer_dq_threshold}): {best}")

    out = {"date": "2026-05-27", "kind": "zeroshot_cross_engine_AB", "n_sentences": len(SENTS),
           "sentences": SENTS, "ceiling": ceiling, "real_utmos_heldout": real_utmos,
           "weights": weights, "wer_dq_threshold": cfg.wer_dq_threshold,
           "naturalness_model": "speechmos:utmos22_strong", "asr_model": "whisperx:large-v3",
           "held_out": [p.name for p in HELD_OUT_REAL],
           "summary": {e: {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in m.items()} for e, m in summary.items()},
           "per_clip": per_clip, "best": best}
    (ROOT / "experiments/ab_eval_results.json").write_text(json.dumps(out, indent=2))

    md = [f"# Zero-shot cross-engine A/B (N={len(SENTS)} sentences/engine)", "",
          f"Real-vs-real ceiling (same-session): **{ceiling:.4f}** (85% target = {0.85 * ceiling:.4f}). "
          f"Real danil UTMOS (held-out): {real_utmos:.3f}.  "
          f"Weights sim {weights['similarity']} / nat {weights['naturalness']} / wer {weights['wer']}.",
          "", "| rank | engine | similarity | % ceiling | UTMOS | WER | score |",
          "|---|---|---|---|---|---|---|"]
    for r, (eng, m) in enumerate(sorted(summary.items(), key=lambda kv: -kv[1]["score"]), 1):
        sc = min(m["similarity"] / ceiling, 1.0) if ceiling > 0 else m["similarity"]
        md.append(f"| {r} | {eng} | {m['similarity']:.4f} | {sc * 100:.0f}% | "
                  f"{m['naturalness']:.3f} | {m['wer']:.4f} | {m['score']:.4f} |")
    md += ["", f"**best:** {best}", ""]
    (ROOT / "reports/eval_ab.md").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/eval_ab.md").write_text("\n".join(md) + "\n")
    print("wrote reports/eval_ab.md + experiments/ab_eval_results.json")


if __name__ == "__main__":
    main()
