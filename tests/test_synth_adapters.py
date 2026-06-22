import numpy as np
import pytest
from voxclone.synth.xtts import XTTSAdapter
from voxclone.synth.f5 import F5Adapter
from voxclone.synth.chatterbox import ChatterboxAdapter


@pytest.mark.parametrize("cls", [XTTSAdapter, F5Adapter, ChatterboxAdapter])
def test_adapter_delegates_to_seam(cls):
    seen = {}

    def fake(text, ref_path, ref_text, params):
        seen.update(text=text, ref=ref_path, rt=ref_text, p=params)
        return np.ones(2048, dtype="float32"), 24000

    a = cls(_synth=fake)
    a.load()  # seam => no real model
    audio, sr = a.synthesize("hi", "/r.wav", ref_text="ref", params={"temperature": 0.6})
    assert sr == 24000 and audio.dtype == np.float32 and audio.shape == (2048,)
    assert seen["text"] == "hi" and seen["rt"] == "ref" and seen["p"]["temperature"] == 0.6
