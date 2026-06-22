import numpy as np
from voxclone.prep.transcribe import transcribe_clip, Transcription

class FakeModel:
    def transcribe(self, audio, sr):
        return [
            {"word": "Hello", "probability": 0.9},
            {"word": "world", "probability": 0.7},
        ]

def test_transcribe_joins_words_and_averages_confidence():
    out = transcribe_clip(np.zeros(48000, dtype=np.float32), 48000, FakeModel())
    assert isinstance(out, Transcription)
    assert out.text == "Hello world"
    assert abs(out.confidence - 0.8) < 1e-6

def test_empty_transcription_is_zero_confidence():
    class Empty:
        def transcribe(self, audio, sr):
            return []
    out = transcribe_clip(np.zeros(10, dtype=np.float32), 48000, Empty())
    assert out.text == ""
    assert out.confidence == 0.0

def test_transcribe_ignores_blank_word_tokens():
    class WithBlank:
        def transcribe(self, audio, sr):
            return [{"word": "hi", "probability": 0.9},
                    {"word": "   ", "probability": 0.1},
                    {"word": "there", "probability": 0.9}]
    import numpy as np
    out = transcribe_clip(np.zeros(10, dtype=np.float32), 48000, WithBlank())
    assert out.text == "hi there"
    assert abs(out.confidence - 0.9) < 1e-6
