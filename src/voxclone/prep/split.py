import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from voxclone.prep.manifest import ClipRecord

@dataclass
class SplitResult:
    train: list[ClipRecord] = field(default_factory=list)
    held_out: list[ClipRecord] = field(default_factory=list)
    enrollment: list[ClipRecord] = field(default_factory=list)

def _words(text: str) -> set[str]:
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def split_dataset(records: list[ClipRecord], *, n_heldout: int = 20, n_enrollment: int = 4,
                  seed: int = 0, enrollment_min_s: float = 6.0, enrollment_max_s: float = 10.0,
                  dup_threshold: float = 0.8) -> SplitResult:
    """Partition a manifest into enrollment (clean, in duration window),
    held-out (~n_heldout, spanning categories, de-duplicated), and train (rest).
    Near-duplicate texts never straddle the train/held-out boundary. Deterministic per seed."""
    rng = random.Random(seed)
    recs = list(records)

    enroll_pool = sorted(
        (r for r in recs if enrollment_min_s <= r.duration <= enrollment_max_s),
        key=lambda r: (r.snr_db, r.transcript_confidence, r.audio_path), reverse=True)
    enrollment = enroll_pool[:n_enrollment]
    enroll_ids = {r.audio_path for r in enrollment}
    remaining = [r for r in recs if r.audio_path not in enroll_ids]

    words = {r.audio_path: _words(r.text) for r in remaining}
    by_cat: dict[str, list[ClipRecord]] = {}
    for r in remaining:
        by_cat.setdefault(r.category, []).append(r)
    for c in by_cat:
        rng.shuffle(by_cat[c])

    held_out: list[ClipRecord] = []
    held_words: list[set[str]] = []
    cats = sorted(by_cat)
    pos = {c: 0 for c in cats}
    while len(held_out) < min(n_heldout, len(remaining)):
        progressed = False
        for c in cats:
            if len(held_out) >= n_heldout:
                break
            lst = by_cat[c]
            while pos[c] < len(lst):
                r = lst[pos[c]]; pos[c] += 1
                ws = words[r.audio_path]
                if any(_jaccard(ws, hw) >= dup_threshold for hw in held_words):
                    continue
                held_out.append(r); held_words.append(ws); progressed = True
                break
        if not progressed:
            break
    held_ids = {r.audio_path for r in held_out}

    train: list[ClipRecord] = []
    for r in remaining:
        if r.audio_path in held_ids:
            continue
        if any(_jaccard(words[r.audio_path], hw) >= dup_threshold for hw in held_words):
            continue
        train.append(r)

    return SplitResult(train=train, held_out=held_out, enrollment=enrollment)

def write_heldout_tsv(path: str | Path, held_out: list[ClipRecord]) -> None:
    """Eval-harness held-out format consumed by scripts/eval.py: text<TAB>real_clip."""
    lines = [f"{r.text}\t{r.audio_path}" for r in held_out]
    Path(path).write_text("\n".join(lines), encoding="utf-8")

def write_enrollment(path: str | Path, enrollment: list[ClipRecord]) -> None:
    """Enrollment clips + transcripts (path<TAB>text) for synth conditioning references."""
    lines = [f"{r.audio_path}\t{r.text}" for r in enrollment]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
