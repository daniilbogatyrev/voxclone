import inspect

from voxclone.train import TRAIN_ENGINES
from voxclone.train.gptsovits import GPTSoVITSTrainer
from voxclone.train.xtts import XTTSTrainer
from voxclone.train.f5 import F5Trainer


def test_train_map_has_three_engines_no_chatterbox():
    assert set(TRAIN_ENGINES) == {"xtts", "f5", "gptsovits"}
    assert "chatterbox" not in TRAIN_ENGINES   # zero-shot only; never fine-tuned
    assert TRAIN_ENGINES["xtts"] is XTTSTrainer
    assert TRAIN_ENGINES["f5"] is F5Trainer
    assert TRAIN_ENGINES["gptsovits"] is GPTSoVITSTrainer


def test_train_values_are_classes_with_train_method():
    for name, cls in TRAIN_ENGINES.items():
        assert inspect.isclass(cls), f"{name} must map to a class"
        train = getattr(cls, "train", None)
        assert callable(train), f"{name} class must expose a callable train()"
        # train(self, manifest_path, out_dir, config) -> TrainResult
        params = list(inspect.signature(train).parameters)
        assert params[:4] == ["self", "manifest_path", "out_dir", "config"], (
            f"{name}.train has unexpected signature: {params}"
        )
