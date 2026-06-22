def combined_score(sim: float, utmos: float, wer: float, weights: dict,
                   ceiling: float | None = None) -> float:
    # Express similarity as a fraction of the real-vs-real ceiling (capped at 1.0) when known,
    # so "1.0 == indistinguishable from the real speaker". Without a ceiling, use sim as-is.
    sim_eff = sim if not ceiling or ceiling <= 0 else min(sim / ceiling, 1.0)
    return (weights["similarity"] * sim_eff
            + weights["naturalness"] * (utmos / 5.0)
            + weights["wer"] * (1.0 - wer))

def pick_best(metrics: dict[str, dict], weights: dict,
              wer_dq_threshold: float | None = None, ceiling: float | None = None) -> str:
    def score(m: dict) -> float:
        return combined_score(m["similarity"], m["naturalness"], m["wer"], weights, ceiling=ceiling)
    # Intelligibility is a prerequisite, not a tradeable axis: drop candidates above the WER floor.
    eligible = {n: m for n, m in metrics.items()
                if wer_dq_threshold is None or m["wer"] <= wer_dq_threshold}
    pool = eligible or metrics  # if everyone is disqualified, fall back to the full set
    return max(pool, key=lambda n: score(pool[n]))
