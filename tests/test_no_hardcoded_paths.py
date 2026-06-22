"""Portability guard: runtime code + launch scripts must not hardcode an absolute
user path (``/home/prada/...``), or a clone under any other path/user/home breaks
(see docs/SETUP.md). Paths must derive from the repo root (``Path(__file__)`` /
``$BASH_SOURCE``) or ``$HOME``."""
import inspect
from pathlib import Path

import voxclone.synth.xtts as xtts
import voxclone.synth.f5 as f5
import voxclone.train.gptsovits as gptsovits

REPO = Path(__file__).resolve().parents[1]


def test_engine_modules_have_no_hardcoded_user_path():
    for mod in (xtts, f5, gptsovits):
        assert "/home/prada" not in inspect.getsource(mod), (
            f"{mod.__name__} hardcodes /home/prada — derive from Path(__file__)/$HOME instead")


def test_scripts_and_notebooks_have_no_hardcoded_user_path():
    for rel in ("scripts/serve_engines.sh", "scripts/serve_finetuned.sh",
                "notebooks/build_voice_cloning_study.py", "src/voxclone/serve/engine_server.py"):
        assert "/home/prada" not in (REPO / rel).read_text(), f"{rel} hardcodes /home/prada"


def test_xtts_model_dir_is_repo_relative():
    assert Path(xtts.MODEL_DIR) == REPO / "third_party" / "xtts_v2_model"


def test_f5_german_paths_are_repo_relative():
    assert Path(f5.GERMAN_CKPT) == REPO / "third_party" / "f5_tts_german" / "F5TTS_Base" / "model_365000.safetensors"
    assert Path(f5.GERMAN_VOCAB) == REPO / "third_party" / "f5_tts_german" / "vocab.txt"
