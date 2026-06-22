from typing import Protocol
from dataclasses import dataclass

@dataclass
class TrainResult:
    checkpoint_dir: str
    steps: int

class TrainAdapter(Protocol):
    def train(self, manifest_path: str, out_dir: str, config: dict) -> TrainResult: ...
