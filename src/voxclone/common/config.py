from pathlib import Path
import yaml
from pydantic import BaseModel

class ValidationConfig(BaseModel):
    min_duration_s: float = 3.0
    max_duration_s: float = 11.0
    max_clipped_fraction: float = 0.005
    min_transcript_confidence: float = 0.6
    min_snr_db: float = 30.0
    max_text_chars: int = 200

class CaptureConfig(BaseModel):
    sample_rate: int = 48000
    target_minutes: float = 50.0

class PrepConfig(BaseModel):
    target_sample_rate: int = 48000
    vad_min_duration_s: float = 3.0
    vad_max_duration_s: float = 11.0
    denoise_enabled: bool = True
    peak_dbfs: float = -1.0
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    validation: ValidationConfig = ValidationConfig()

class EvalConfig(BaseModel):
    similarity: float = 0.30
    naturalness: float = 0.50
    wer: float = 0.20
    wer_dq_threshold: float = 0.20
    target_lufs: float = -23.0
    eval_sr: int = 16000

    def weights(self) -> dict:
        return {"similarity": self.similarity, "naturalness": self.naturalness, "wer": self.wer}

class AppConfig(BaseModel):
    capture: CaptureConfig = CaptureConfig()
    prep: PrepConfig = PrepConfig()
    eval: EvalConfig = EvalConfig()

def load_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return AppConfig(**data)
