import json
from pathlib import Path

class ModelRegistry:
    def __init__(self, runs_dir: str | Path):
        self.path = Path(runs_dir) / "registry.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def register(self, model: str, checkpoint: str, metrics: dict) -> None:
        data = self._load()
        data.setdefault(model, []).append({"checkpoint": checkpoint, "metrics": metrics})
        self.path.write_text(json.dumps(data, indent=2))

    def best_checkpoint(self, model: str) -> str | None:
        entries = self._load().get(model, [])
        if not entries:
            return None
        best = max(entries, key=lambda e: e["metrics"].get("score", float("-inf")))
        return best["checkpoint"]

    def best_for_engine(self, engine: str) -> tuple[str, str] | None:
        """Resolve the eval-registered WINNER for a SHORT engine key.

        eval registers each candidate under the COMPOUND key ``<engine>_<label>``
        (e.g. ``xtts_finetuned``), but serve resolves by the bare engine key (``xtts``).
        This scans every registered candidate whose key is ``engine`` itself OR starts
        with ``engine + "_"`` (the engine's label variants) and returns the
        ``(key, checkpoint)`` of the highest-``score`` one, or ``None`` if the engine has
        no candidates.
        """
        data = self._load()
        best_key, best_ckpt, best_score = None, None, float("-inf")
        for key, entries in data.items():
            if key != engine and not key.startswith(engine + "_"):
                continue
            for entry in entries:
                score = entry["metrics"].get("score", float("-inf"))
                if score > best_score:
                    best_key, best_ckpt, best_score = key, entry["checkpoint"], score
        if best_key is None:
            return None
        return best_key, best_ckpt
