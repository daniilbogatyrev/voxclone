import csv
import json
from pathlib import Path
from pydantic import BaseModel

class ClipRecord(BaseModel):
    audio_path: str
    text: str
    duration: float
    transcript_confidence: float
    clipped_fraction: float
    category: str
    snr_db: float = 60.0

def write_manifest(path: str | Path, records: list[ClipRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

def read_manifest(path: str | Path) -> list[ClipRecord]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ClipRecord(**json.loads(line)))
    return records

def write_transcript_csv(path: str | Path, records: list[ClipRecord]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "category", "transcript_confidence", "text"])
        for r in records:
            writer.writerow([r.audio_path, r.category,
                             f"{r.transcript_confidence:.4g}", r.text])
