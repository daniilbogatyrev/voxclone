from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_finetuned.sh"


def test_script_is_executable_shaped():
    text = SCRIPT.read_text()
    assert text.startswith("#!"), "missing shebang"


def test_launches_both_finetuned_checkpoint_servers():
    text = SCRIPT.read_text()
    # The eval/serve checkpoint-bound adapters POST to xtts_server (9881) and
    # f5_server (9882) -- NOT the uniform engine_server.
    for token in [
        "voxclone.serve.xtts_server", "voxclone.serve.f5_server",
        "9881", "9882",
        "conda run",
        "-n xtts", "-n f5tts",
        "COQUI_TOS_AGREED",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
    ]:
        assert token in text, token
