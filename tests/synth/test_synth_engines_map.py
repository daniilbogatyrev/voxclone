"""SYNTH_ENGINES name->class map (the eval/serve-side naming contract).

Keys are the SHORT engine names {xtts, f5, gptsovits, chatterbox} -- the load-bearing
engine name (== serve model name == registry key == --out basename == eval candidate
engine token), distinct from the notebook client.ENGINES LONG keys.

The *Synth values are the eval/serve-side names of the P02 adapter classes
(XTTSSynth == XTTSAdapter, etc.; GPTSoVITSSynth is already that name). Each is a class
exposing `synthesize`. Importing the map must NOT pull torch / TTS (heavy engine libs are
imported lazily inside the real generate path, not at module import time).
"""
import sys

from voxclone.synth import (
    SYNTH_ENGINES,
    XTTSSynth,
    F5Synth,
    GPTSoVITSSynth,
    ChatterboxSynth,
)


def test_synth_map_has_four_short_engine_keys_mapped_to_classes():
    assert SYNTH_ENGINES == {
        "xtts": XTTSSynth,
        "f5": F5Synth,
        "gptsovits": GPTSoVITSSynth,
        "chatterbox": ChatterboxSynth,
    }
    assert set(SYNTH_ENGINES) == {"xtts", "f5", "gptsovits", "chatterbox"}


def test_each_value_is_a_class_exposing_synthesize():
    for name, cls in SYNTH_ENGINES.items():
        assert isinstance(cls, type), f"{name} value is not a class"
        assert callable(getattr(cls, "synthesize", None)), f"{name} class lacks synthesize"


def test_short_keys_distinct_from_client_long_keys():
    # The notebook client uses LONG keys; this map must use the SHORT naming contract.
    long_keys = {"gptsovits_v2pro", "xtts_v2", "f5_tts"}
    assert long_keys.isdisjoint(SYNTH_ENGINES)


def test_map_import_does_not_pull_torch_or_tts():
    # Fresh-import voxclone.synth and assert no heavy engine lib was imported as a side effect.
    for mod in [m for m in list(sys.modules) if m == "voxclone.synth" or m.startswith("voxclone.synth.")]:
        del sys.modules[mod]
    for heavy in ("torch", "TTS", "f5_tts", "chatterbox"):
        sys.modules.pop(heavy, None)

    import importlib

    importlib.import_module("voxclone.synth")

    for heavy in ("torch", "TTS", "f5_tts", "chatterbox"):
        assert heavy not in sys.modules, f"importing voxclone.synth pulled heavy dep {heavy}"
