from typing import Protocol
import numpy as np

class SynthAdapter(Protocol):
    def synthesize(self, text: str, reference_clip: str,
                   params: dict) -> tuple[np.ndarray, int]: ...
