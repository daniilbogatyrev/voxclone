from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_engines.sh"


def test_launch_script_covers_all_engines():
    text = SCRIPT.read_text()
    for token in ["--engine xtts", "--engine f5", "--engine chatterbox",
                  "9881", "9882", "9883", "api_v2.py", "LD_LIBRARY_PATH",
                  "PYTHONPATH", "conda run"]:
        assert token in text, token
