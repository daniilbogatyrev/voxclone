import numpy as np
from voxclone.common.audio import save_audio
from voxclone.common.config import PrepConfig
from voxclone.prep.pipeline import run_prep

def _speechlike(sr, seconds=10):
    rng = np.random.default_rng(0)
    blocks = []
    for k in range(seconds):
        if k % 2 == 0:
            blocks.append((0.3*np.sin(2*np.pi*180*np.arange(sr)/sr)).astype(np.float32))
        else:
            blocks.append((0.001*rng.standard_normal(sr)).astype(np.float32))
    return np.concatenate(blocks)

def _segmenter(audio, sr, vad):
    return [{"start": 0.0, "end": len(audio)/sr}]

class _T:
    def transcribe(self, audio, sr):
        return [{"word": "smoke", "probability": 0.9},
                {"word": "test", "probability": 0.9}]

def test_end_to_end_pipeline_cpu(tmp_path):
    raw = tmp_path/"raw"; raw.mkdir()
    sr = 48000
    for i in range(2):
        save_audio(raw/f"freeform_{i:04d}.wav", _speechlike(sr), sr)
    result = run_prep(raw, tmp_path/"proc", PrepConfig(denoise_enabled=False),
                      transcriber=_T(), segmenter=_segmenter,
                      manifest_path=tmp_path/"m.jsonl",
                      transcript_csv=tmp_path/"t.csv",
                      quarantine_report=tmp_path/"q.json")
    assert len(result.kept) >= 2
    assert (tmp_path/"m.jsonl").read_text().count("\n") == len(result.kept)
