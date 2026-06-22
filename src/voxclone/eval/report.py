from pathlib import Path

def render_markdown(rows: list[dict], ceiling: float) -> str:
    lines = [
        f"# Eval report",
        "",
        f"Real-vs-real speaker-similarity ceiling: **{ceiling:.4f}** "
        f"(success target = 85% = {0.85 * ceiling:.4f}).",
        "",
        "| checkpoint | similarity | naturalness | wer | score |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['checkpoint']} | {r['similarity']:.4f} | {r['naturalness']:.3f} "
            f"| {r['wer']:.4f} | {r['score']:.4f} |"
        )
    return "\n".join(lines) + "\n"

def write_report(path: str | Path, rows: list[dict], ceiling: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_markdown(rows, ceiling), encoding="utf-8")
