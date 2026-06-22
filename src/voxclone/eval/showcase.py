"""Render helpers for the interactive bake-off notebook: turn ab_eval_results.json +
gallery audio into portable HTML players + matplotlib figures. Replaces the never-built
eval.compare. NO model loading here — Phase 1 is GPU-free (live scoring lives in Plan 03)."""
import base64
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")  # headless: figures render the same in tests + notebooks
import matplotlib.pyplot as plt

from voxclone.common.audio import load_audio
from voxclone.eval.normalize import normalize_for_eval
from voxclone.eval.similarity import cosine, reference_ceiling
from voxclone.eval.wer import wer as _wer
from voxclone.prep.transcribe import transcribe_clip

ENGINE_LABELS = {
    "f5_tts": "F5-TTS",
    "xtts_v2": "XTTS-v2",
    "chatterbox": "Chatterbox",
    "gptsovits_v2pro": "GPT-SoVITS v2Pro",
}
# dark palette echoing frontend/style.css for a cohesive portfolio look
BG, FG, ACCENT, MUTED = "#11131a", "#e6e8ef", "#4f7cff", "#8b90a3"


def load_ab_results(path="experiments/ab_eval_results.json") -> dict:
    return json.loads(Path(path).read_text())


def embed_audio(path, target_sr: int = 22050) -> str:
    """Load audio, downmix mono, resample, return a base64 WAV data URI for an inline <audio>."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("float32")
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype("float32")
        sr = target_sr
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _chip(label, value_html):
    return ('<div style="font-size:11px;color:%s;letter-spacing:-.02em">%s '
            '<span style="color:%s">%s</span></div>' % (MUTED, label, FG, value_html))


def render_gallery_html(gallery: dict, gallery_dir) -> str:
    """sentences (hero subset) x engines grid: embedded players + metric chips + ★ winner.
    Uses %-formatting (not f-strings) to stay clear of nested-quote issues on Python 3.11."""
    gallery_dir = Path(gallery_dir)
    engines = list(ENGINE_LABELS)
    ceiling = gallery["ceiling"]
    by_engine = gallery["per_clip"]
    sent = {i + 1: s for i, s in enumerate(gallery["sentences"])}
    head = "".join(
        '<th style="padding:8px;text-align:left;color:%s">%s%s</th>'
        % (FG, ENGINE_LABELS[e], " ★" if e == gallery["best"] else "")
        for e in engines)
    rows = []
    for idx in gallery["hero_indices"]:
        tds = []
        for e in engines:
            m = next(c for c in by_engine[e] if c["i"] == idx)
            uri = embed_audio(gallery_dir / e / ("s%02d.wav" % idx))
            pct = (min(m["sim"] / ceiling, 1.0) * 100) if ceiling else (m["sim"] * 100)
            border = ("2px solid " + ACCENT) if e == gallery["best"] else "1px solid #2a2d3a"
            bar = ('<div style="height:6px;border-radius:3px;background:#2a2d3a;margin-top:4px">'
                   '<div style="height:6px;border-radius:3px;width:%.0f%%;background:%s"></div>'
                   '</div>' % (pct, ACCENT))
            td = ('<td style="padding:8px;border:%s;border-radius:10px;vertical-align:top">'
                  '<audio controls preload="none" src="%s" style="width:160px"></audio>'
                  '%s%s%s%s'
                  '<div title="%s" style="font-size:10px;color:%s;margin-top:4px;max-width:160px;'
                  'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">“%s”</div></td>'
                  % (border, uri,
                     _chip("sim", "%.0f%% of ceiling" % pct), bar,
                     _chip("UTMOS", "%.2f" % m["utmos"]),
                     _chip("WER", "%.0f%%" % (m["wer"] * 100)),
                     m["hyp"], MUTED, m["hyp"]))
            tds.append(td)
        rows.append('<tr><td style="padding:8px;color:%s;max-width:180px">“%s”</td>%s</tr>'
                    % (MUTED, sent[idx], "".join(tds)))
    return ('<div style="background:%s;color:%s;padding:16px;border-radius:12px;'
            'font-family:system-ui;letter-spacing:-.02em">'
            '<table style="border-collapse:separate;border-spacing:8px">'
            '<tr><th></th>%s</tr>%s</table></div>' % (BG, FG, head, "".join(rows)))


def _dark(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=FG)
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def _sorted(summary):
    return sorted(summary.items(), key=lambda kv: -kv[1]["score"])


def plot_leaderboard(res):
    s = _sorted(res["summary"])
    names = [ENGINE_LABELS[n] for n, _ in s]
    scores = [m["score"] for _, m in s]
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = [ACCENT if i == 0 else MUTED for i in range(len(s))]
    ax.barh(names, scores, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("composite score (sim·.30 + nat·.50 + wer·.20)")
    ax.set_title("Bake-off leaderboard")
    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, f"{v:.3f}", color=FG, va="center")
    _dark(ax, fig)
    fig.tight_layout()
    return fig


def plot_similarity_vs_ceiling(res):
    s = _sorted(res["summary"])
    names = [ENGINE_LABELS[n] for n, _ in s]
    sims = [m["similarity"] for _, m in s]
    errs = [m.get("sim_std", 0) for _, m in s]
    c = res["ceiling"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(names, sims, xerr=errs, color=ACCENT, ecolor=MUTED, capsize=3)
    ax.invert_yaxis()
    ax.axvline(c, ls="--", color=FG, label=f"real-vs-real ceiling ({c:.3f})")
    ax.axvline(0.85 * c, ls=":", color=MUTED, label=f"85% target ({0.85*c:.3f})")
    ax.set_xlabel("speaker similarity (ECAPA cosine)")
    ax.set_title("Similarity vs the same-session ceiling")
    ax.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=8)
    _dark(ax, fig)
    fig.tight_layout()
    return fig


def plot_utmos(res):
    s = _sorted(res["summary"])
    names = [ENGINE_LABELS[n] for n, _ in s]
    mos = [m["naturalness"] for _, m in s]
    errs = [m.get("utmos_std", 0) for _, m in s]
    real = res["real_utmos_heldout"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(names, mos, yerr=errs, color=ACCENT, ecolor=MUTED, capsize=3)
    ax.axhline(real, ls="--", color=FG, label=f"real danil ({real:.2f})")
    ax.set_ylim(1, 5)
    ax.set_ylabel("UTMOS (1–5)")
    ax.set_title("Naturalness — clones score above the real recording (clean-audio quirk)")
    ax.legend(facecolor=BG, edgecolor=MUTED, labelcolor=FG, fontsize=8)
    ax.tick_params(axis="x", rotation=15)
    _dark(ax, fig)
    fig.tight_layout()
    return fig


def load_reference_centroid(paths, embedder):
    """Embed real reference clips -> (centroid vector, real-vs-real ceiling)."""
    embs = []
    for p in paths:
        audio, sr = load_audio(p)
        norm, _ = normalize_for_eval(audio, sr)
        embs.append(np.asarray(embedder(norm)))
    arr = np.stack(embs)
    return arr.mean(axis=0), reference_ceiling(arr)


def score_clip(audio, sr, intended_text, *, embedder, transcriber, utmos,
               target_emb, ceiling=None, language="en") -> dict:
    """Score one freshly-generated clip live. `intended_text` is what the engine was asked to
    say (for WER). `target_emb` is the speaker centroid to compare against (danil's, or an
    uploaded reference's). With no ceiling (custom upload), similarity is reported raw.
    `language` selects the WER text normalizer (e.g. "de" preserves umlauts); the ASR language
    is the caller's responsibility (pass a transcriber built for that language)."""
    norm, nsr = normalize_for_eval(audio, sr)
    emb = np.asarray(embedder(norm))
    sim = cosine(emb, np.asarray(target_emb))
    hyp = transcribe_clip(norm, nsr, transcriber).text
    w = _wer(intended_text, hyp, language=language)
    nat = float(utmos(norm, nsr))
    sim_pct = (min(sim / ceiling, 1.0) if ceiling else sim)
    return {"similarity": sim, "sim_pct": sim_pct, "naturalness": nat, "wer": w, "hyp": hyp}
