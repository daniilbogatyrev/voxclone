from functools import lru_cache

@lru_cache(maxsize=2)
def _normalizer(language: str = "en"):
    # English: lowercases, strips punctuation, expands contractions, and converts
    # spelled-out numbers/currency to a canonical form -> meaningful English word WER.
    # Other languages: the basic normalizer only lowercases + strips punctuation, so it
    # PRESERVES diacritics (ä/ö/ü/ß) and skips English-specific rewrites (e.g. "Dr."->"doctor",
    # "3 Euro"->"€3") that would corrupt non-English WER.
    if language == "en":
        from whisper_normalizer.english import EnglishTextNormalizer
        return EnglishTextNormalizer()
    from whisper_normalizer.basic import BasicTextNormalizer
    return BasicTextNormalizer()

def _normalize(text: str, language: str = "en") -> list[str]:
    return _normalizer(language)(text).split()

def wer(reference: str, hypothesis: str, language: str = "en") -> float:
    ref = _normalize(reference, language)
    hyp = _normalize(hypothesis, language)
    if not ref:
        return 0.0 if not hyp else 1.0
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return min(d[len(ref)][len(hyp)] / len(ref), 1.0)
