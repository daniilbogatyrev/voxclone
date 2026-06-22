"""Synth package: the SYNTH_ENGINES name->class map (eval/serve naming contract).

The dict KEY is the load-bearing SHORT engine name -- identical across the whole
train -> registry -> serve/eval chain:

    SYNTH_ENGINES key == serve model name == registry key == --out basename
                      == eval candidate engine token

These SHORT keys {xtts, f5, gptsovits, chatterbox} are deliberately DISTINCT from the
notebook `client.ENGINES` LONG keys {xtts_v2, f5_tts, gptsovits_v2pro, chatterbox}.

The *Synth values are the eval/serve-facing, CHECKPOINT-BOUND adapters: each takes
``(checkpoint, generate_fn)`` at construction and exposes ``synthesize(text, reference_clip,
params)``, because ``eval/runner.py`` constructs them as ``SYNTH_ENGINES[engine](checkpoint=...)``
and calls ``synthesize(text, ref, {})`` with EMPTY params. These are DISTINCT from the
notebook-facing P02 ``*Adapter`` classes (``XTTSAdapter`` etc., ctor ``(model_dir/device,
_synth)``) which the bake-off ``engine_server`` uses -- do not alias one for the other.

These imports are light: every engine's heavy lib (torch / TTS / f5_tts / chatterbox) is
imported lazily inside the adapter's real generate path, never at module import time, so
`from voxclone.synth import SYNTH_ENGINES` stays GPU-free.
"""
from voxclone.synth.gptsovits import GPTSoVITSSynth
from voxclone.synth.xtts import XTTSSynth
from voxclone.synth.f5 import F5Synth
from voxclone.synth.chatterbox import ChatterboxSynth

# Engine name -> SynthAdapter class. The key is the load-bearing engine name.
SYNTH_ENGINES = {
    "xtts": XTTSSynth,
    "f5": F5Synth,
    "gptsovits": GPTSoVITSSynth,
    "chatterbox": ChatterboxSynth,
}

__all__ = [
    "SYNTH_ENGINES",
    "XTTSSynth",
    "F5Synth",
    "GPTSoVITSSynth",
    "ChatterboxSynth",
]
