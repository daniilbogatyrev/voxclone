from voxclone.common.registry import ModelRegistry

def test_register_and_resolve_best_by_score(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register("gptsovits", "/ckpt/a", {"score": 0.7})
    reg.register("gptsovits", "/ckpt/b", {"score": 0.9})
    reg.register("gptsovits", "/ckpt/c", {"score": 0.8})
    assert reg.best_checkpoint("gptsovits") == "/ckpt/b"

def test_unknown_model_returns_none(tmp_path):
    assert ModelRegistry(tmp_path).best_checkpoint("nope") is None

def test_registry_persists_across_instances(tmp_path):
    ModelRegistry(tmp_path).register("m", "/ckpt/x", {"score": 0.5})
    assert ModelRegistry(tmp_path).best_checkpoint("m") == "/ckpt/x"
