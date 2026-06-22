"""Evaluate every bake-off candidate through the Plan 06 fairness harness.

Each ``--candidates`` token is ``engine:label=checkpoint`` (e.g.
``xtts:finetuned=runs/xtts`` or ``chatterbox:zeroshot=base``). The ``engine`` half
selects the SynthAdapter class from ``voxclone.synth.SYNTH_ENGINES`` (NOT a hardcoded
GPTSoVITSSynth) and the ``=checkpoint`` half binds its weights; ``label`` distinguishes
``finetuned`` vs ``zeroshot`` in the report. The candidate key (``<engine>_<label>``) is
the registry model key written after scoring, so ``serve`` can resolve ``best_checkpoint``.

Scoring weights and the WER disqualification threshold come from ``EvalConfig``
(``weights()`` -> 0.30 / 0.50 / 0.20 and ``wer_dq_threshold`` -> 0.20), replacing the
old hardcoded 0.5 / 0.3 / 0.2 literals. Pass ``--config`` to override the defaults.
"""
import argparse
from pathlib import Path

from voxclone.common.config import load_config, AppConfig
from voxclone.common.registry import ModelRegistry
from voxclone.eval.naturalness import load_speechmos_utmos
from voxclone.eval.runner import run_eval
from voxclone.eval.similarity import load_ecapa
from voxclone.prep.transcribe import load_transcriber
from voxclone.synth import SYNTH_ENGINES


def _parse_candidate(token: str) -> tuple[str, str, str]:
    """Split ``engine:label=checkpoint`` into ``(name, engine, checkpoint)``.

    ``name`` is the registry/report key ``<engine>_<label>``; ``engine`` selects the
    synth class. Raises ValueError on a malformed token or unknown engine.
    """
    cand, sep, checkpoint = token.partition("=")
    if not sep:
        raise ValueError(f"candidate {token!r} must be engine:label=checkpoint")
    engine, sep, label = cand.partition(":")
    if not sep:
        raise ValueError(f"candidate {token!r} must be engine:label=checkpoint")
    if engine not in SYNTH_ENGINES:
        raise ValueError(f"unknown engine {engine!r}; synthesizable: {sorted(SYNTH_ENGINES)}")
    return f"{engine}_{label}", engine, checkpoint


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="evaluate bake-off candidates")
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="engine:label=checkpoint tokens (e.g. xtts:finetuned=runs/xtts)")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ref-text", default="",
                    help="transcript of --reference (F5 conditions on it as prompt_text); "
                         "ignored by engines that do not use it")
    ap.add_argument("--held-out", required=True, help="tsv: text<TAB>real_clip")
    ap.add_argument("--report", default="reports/eval.md")
    ap.add_argument("--register", default=None,
                    help="runs dir for the ModelRegistry (registry.json); omit to skip persistence")
    ap.add_argument("--config", default=None, help="app config yaml (for EvalConfig weights/threshold)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    held = [tuple(line.split("\t")) for line in
            Path(args.held_out).read_text().splitlines() if line]

    # Each candidate's synth class is selected per-engine from SYNTH_ENGINES; the candidate
    # name (engine_label) is the registry key serve later resolves best_checkpoint over.
    checkpoints = {}
    for token in args.candidates:
        name, engine, checkpoint = _parse_candidate(token)
        checkpoints[name] = SYNTH_ENGINES[engine](checkpoint=checkpoint)

    cfg = load_config(args.config).eval if args.config else AppConfig().eval

    registry = ModelRegistry(args.register) if args.register else None

    result = run_eval(checkpoints, held, args.reference, load_ecapa(),
                      load_transcriber("large-v3", args.device), load_speechmos_utmos(),
                      cfg.weights(), args.report,
                      wer_dq_threshold=cfg.wer_dq_threshold, registry=registry,
                      ref_text=args.ref_text)
    print(f"best: {result['best']} (ceiling {result['ceiling']:.4f})")


if __name__ == "__main__":  # pragma: no cover
    main()
