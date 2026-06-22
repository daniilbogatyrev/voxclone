def test_notebook_deps_import():
    import nbformat  # noqa: F401
    import matplotlib  # noqa: F401
    import IPython  # noqa: F401


import numpy as np
import soundfile as sf
from voxclone.eval import showcase


def test_load_ab_results(tmp_path):
    import json
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"ceiling": 0.8, "summary": {}}))
    assert showcase.load_ab_results(p)["ceiling"] == 0.8


def test_embed_audio_returns_wav_data_uri(tmp_path):
    wav = tmp_path / "c.wav"
    sf.write(wav, (0.1 * np.random.randn(48000)).astype("float32"), 48000)
    uri = showcase.embed_audio(wav)
    assert uri.startswith("data:audio/wav;base64,")
    assert len(uri) > 100  # actually carries data


def _sample_gallery(tmp_path):
    import json
    eng = list(showcase.ENGINE_LABELS)
    for e in eng:
        d = tmp_path / e
        d.mkdir()
        sf.write(d / "s01.wav", (0.1 * np.random.randn(22050)).astype("float32"), 22050)
    gal = {
        "hero_indices": [1], "sentences": ["Hello there."], "best": "f5_tts",
        "ceiling": 0.8284,
        "per_clip": {e: [{"i": 1, "sim": 0.7, "wer": 0.0, "utmos": 3.8, "hyp": "hello there"}]
                     for e in eng},
    }
    (tmp_path / "gallery.json").write_text(json.dumps(gal))
    return gal


def test_render_gallery_html(tmp_path):
    gal = _sample_gallery(tmp_path)
    html = showcase.render_gallery_html(gal, tmp_path)
    assert "<audio" in html and "data:audio/wav;base64," in html  # embedded players
    assert "F5-TTS" in html                                       # engine label
    assert "Hello there." in html                                 # the sentence
    assert "★" in html                                            # winner marked
    assert "%" in html                                            # sim-as-%-of-ceiling chip


def test_charts_return_figures():
    import matplotlib.figure
    summary = {
        "f5_tts": {"similarity": 0.736, "naturalness": 3.865, "wer": 0.008,
                   "score": 0.851, "sim_std": 0.04, "utmos_std": 0.10},
        "gptsovits_v2pro": {"similarity": 0.440, "naturalness": 3.630, "wer": 0.008,
                            "score": 0.721, "sim_std": 0.12, "utmos_std": 0.22},
    }
    res = {"summary": summary, "ceiling": 0.8284, "real_utmos_heldout": 3.295,
           "weights": {"similarity": 0.3, "naturalness": 0.5, "wer": 0.2}}
    for fn in (showcase.plot_leaderboard, showcase.plot_similarity_vs_ceiling,
               showcase.plot_utmos):
        assert isinstance(fn(res), matplotlib.figure.Figure)


def test_load_speechmos_utmos_wraps_model(monkeypatch):
    import types
    import numpy as np
    from voxclone.eval import naturalness

    class _Scalar:
        def __init__(self, v): self.v = v
        def __float__(self): return self.v

    class FakeModel:
        def __call__(self, wav, sr):
            assert sr == 16000
            return _Scalar(4.2)

    fake_hub = types.SimpleNamespace(load=lambda *a, **k: FakeModel())
    monkeypatch.setattr(naturalness, "_torch_hub", lambda: fake_hub)
    score = naturalness.load_speechmos_utmos()
    assert abs(score(np.zeros(16000, dtype="float32"), 16000) - 4.2) < 1e-6


def test_score_clip_with_seams():
    import numpy as np
    from voxclone.eval import showcase

    embedder = lambda a: np.array([1.0, 0.0, 0.0])

    class T:
        def transcribe(self, a, sr):
            return [{"word": "hello", "probability": 0.9}, {"word": "world", "probability": 0.9}]

    utmos = lambda a, sr: 4.1
    out = showcase.score_clip(
        np.zeros(16000, dtype="float32"), 16000, "hello world",
        embedder=embedder, transcriber=T(), utmos=utmos,
        target_emb=np.array([1.0, 0.0, 0.0]), ceiling=0.8)
    assert out["wer"] == 0.0
    assert abs(out["naturalness"] - 4.1) < 1e-6
    assert abs(out["similarity"] - 1.0) < 1e-6
    assert abs(out["sim_pct"] - 1.0) < 1e-6  # 1.0 / 0.8 capped at 1.0


def test_score_clip_german_scores_wer_with_german_normalizer():
    import numpy as np
    from voxclone.eval import showcase

    embedder = lambda a: np.array([1.0, 0.0, 0.0])

    class T:  # ASR transcribes the umlaut-less spelling — a real German error
        def transcribe(self, a, sr):
            return [{"word": "Muller", "probability": 0.9}]

    out = showcase.score_clip(
        np.zeros(16000, dtype="float32"), 16000, "Müller",
        embedder=embedder, transcriber=T(), utmos=lambda a, sr: 4.0,
        target_emb=np.array([1.0, 0.0, 0.0]), ceiling=0.8, language="de")
    # German normalizer keeps the umlaut, so "Müller" != "Muller" -> WER 1.0.
    # (Default English normalizer would strip it and wrongly report 0.0.)
    assert out["wer"] == 1.0
