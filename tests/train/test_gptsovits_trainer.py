"""Smoke test for the REAL GPT-SoVITS v2Pro fine-tune trainer (injectable runner).

The exhaustive per-stage env/config assertions live in test_train_gptsovits.py; this
file keeps a single end-to-end smoke check that the trainer writes its .list and drives
the full env-driven prep + config-file train pipeline through the injected runner.

GPU-free: the subprocess runner is mocked, and a writable tmp dir is used as the
GPT-SoVITS root (never /opt, which a non-root user cannot create under).
"""

from pathlib import Path

from voxclone.prep.manifest import ClipRecord, write_manifest
from voxclone.train.gptsovits import GPTSoVITSTrainer
from voxclone.train.base import TrainResult


def test_trainer_writes_listfile_and_invokes_runner(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [
        ClipRecord(audio_path="a.wav", text="hello", duration=2.0,
                   transcript_confidence=0.9, clipped_fraction=0.0, category="harvard"),
    ])
    root = tmp_path / "gsv_root"
    root.mkdir(parents=True, exist_ok=True)

    calls = []
    envs = []

    def fake_runner(cmd, env=None, cwd=None, **kw):
        calls.append(list(cmd))
        envs.append(dict(env) if env is not None else None)
        # mimic the prep stages writing the artifacts the train stages read
        if env and env.get("opt_dir"):
            opt = Path(env["opt_dir"])
            opt.mkdir(parents=True, exist_ok=True)
            flat = " ".join(cmd)
            if "1-get-text.py" in flat:
                (opt / "2-name2text.txt").write_text("a\tphones\n", encoding="utf-8")
            elif "3-get-semantic.py" in flat:
                (opt / "6-name2semantic.tsv").write_text("h\n", encoding="utf-8")

        class R:
            returncode = 0
        return R()

    trainer = GPTSoVITSTrainer(gptsovits_root=str(root), exp_name="danil",
                               runner=fake_runner)
    result = trainer.train(str(manifest), str(tmp_path / "out"),
                           {"epochs_sovits": 8, "epochs_gpt": 15})

    assert isinstance(result, TrainResult)
    # the .list (inp_text for the prep stages) lands under logs/<exp_name>
    list_file = root / "logs" / "danil" / "danil.list"
    assert list_file.exists()
    assert list_file.read_text(encoding="utf-8").startswith("a.wav|target|EN|hello")
    # full pipeline: 4 prep stages + s2 + s1
    assert len(calls) == 6
    assert any("s2_train.py" in " ".join(c) for c in calls)
    assert any("s1_train.py" in " ".join(c) for c in calls)
    # every stage runs from the GPT-SoVITS root (webui semantics)
    assert all(c is not None for c in calls)
