import numpy as np
from pathlib import Path
from voxclone.common.audio import save_audio
from voxclone.common.config import PrepConfig
from voxclone.prep.pipeline import run_prep, PrepResult

def _fake_vad_segments(audio, sr, vad):
    dur = len(audio) / sr
    return [{"start": 0.0, "end": dur}]

class _FakeTranscriber:
    def transcribe(self, audio, sr):
        return [{"word": "hello", "probability": 0.95},
                {"word": "there", "probability": 0.95}]

def _speechlike(sr, seconds=8):
    rng = np.random.default_rng(0)
    blocks = []
    for k in range(seconds):
        if k % 2 == 0:
            blocks.append((0.3 * np.sin(2*np.pi*200*np.arange(sr)/sr)).astype(np.float32))
        else:
            blocks.append((0.001 * rng.standard_normal(sr)).astype(np.float32))
    return np.concatenate(blocks)

def test_run_prep_produces_manifest(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    out = tmp_path / "processed"
    sr = 48000
    save_audio(raw / "harvard_0001.wav", _speechlike(sr), sr)
    cfg = PrepConfig(denoise_enabled=False)
    result = run_prep(
        raw_dir=raw, out_dir=out, config=cfg,
        vad=None, segmenter=_fake_vad_segments,
        transcriber=_FakeTranscriber(),
        manifest_path=tmp_path/"manifest.jsonl",
        transcript_csv=tmp_path/"t.csv",
        quarantine_report=tmp_path/"quarantine.json",
    )
    assert isinstance(result, PrepResult)
    assert len(result.kept) >= 1
    assert (tmp_path/"manifest.jsonl").exists()
    assert all(Path(r.audio_path).exists() for r in result.kept)
    assert result.kept[0].text == "hello there"
    from voxclone.common.audio import estimate_snr_db, load_audio
    seg_audio, seg_sr = load_audio(result.kept[0].audio_path)
    assert abs(result.kept[0].snr_db - estimate_snr_db(seg_audio, seg_sr)) < 1e-4

def test_run_prep_resamples_to_target_and_makes_dirs(tmp_path):
    from voxclone.common.audio import load_audio
    raw = tmp_path / "raw"; raw.mkdir()
    sr_in = 44100
    save_audio(raw / "harvard_0001.wav", _speechlike(sr_in), sr_in)
    cfg = PrepConfig(denoise_enabled=False, target_sample_rate=48000)
    # manifest path in a not-yet-existing subdir to exercise mkdir
    result = run_prep(
        raw_dir=raw, out_dir=tmp_path/"processed", config=cfg,
        vad=None, segmenter=_fake_vad_segments, transcriber=_FakeTranscriber(),
        manifest_path=tmp_path/"nested/sub/manifest.jsonl",
        transcript_csv=tmp_path/"nested/sub/t.csv",
        quarantine_report=tmp_path/"nested/sub/q.json",
    )
    assert len(result.kept) >= 1
    assert (tmp_path/"nested/sub/manifest.jsonl").exists()
    _, out_sr = load_audio(result.kept[0].audio_path)
    assert out_sr == 48000
