"""Serve the demo, dispatching each `model` to its engine's *Synth.

`provider(model)` selects the SynthAdapter class from `voxclone.synth.SYNTH_ENGINES`
(NOT a hardcoded `GPTSoVITSSynth`), binds it to the eval-registered WINNER for that
engine, and caches it. The model name is the SHORT engine key {xtts,f5,gptsovits,
chatterbox}; eval registers candidates under COMPOUND keys (`<engine>_<label>`, e.g.
`xtts_finetuned`), so the provider resolves the engine's highest-score candidate via
`ModelRegistry.best_for_engine(model)`. `create_app` stays engine-agnostic: it just
routes `req.model` through `provider`.
"""
import argparse

import uvicorn

from voxclone.common.registry import ModelRegistry
from voxclone.serve.app import create_app, create_studio_app
from voxclone.synth import SYNTH_ENGINES


def make_provider(reg: ModelRegistry):
    """Return a `provider(model) -> Synth` that builds the right *Synth per model.

    `model` is the SHORT engine key; eval registers candidates under compound keys
    (`<engine>_<label>`), so each model resolves to
    `SYNTH_ENGINES[model](checkpoint=reg.best_for_engine(model)[1])` -- the engine's
    highest-score eval winner -- cached per model. Raises KeyError when the model is an
    unknown engine or has no registered candidate, so `serve.app` maps it to a 404.
    """
    _synth_cache: dict[str, object] = {}

    def provider(model: str):
        if model not in _synth_cache:
            cls = SYNTH_ENGINES.get(model)
            if cls is None:
                raise KeyError(model)
            winner = reg.best_for_engine(model)
            if winner is None:
                raise KeyError(model)
            _synth_cache[model] = cls(checkpoint=winner[1])
        return _synth_cache[model]

    return provider


def main() -> None:
    ap = argparse.ArgumentParser(description="serve the demo")
    ap.add_argument("--studio", action="store_true",
                    help="serve the 5-voice clone studio (English/German) backed by "
                         "voxclone.clone.studio -- the same backend as the notebook. "
                         "No --reference/--runs needed (studio has its own ref + voices).")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--reference", help="reference clip (required unless --studio)")
    ap.add_argument("--ref-text", default="",
                    help="transcript of --reference (F5 conditions on it as prompt_text); "
                         "ignored by engines that do not use it")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.studio:  # pragma: no cover (launches a real server)
        uvicorn.run(create_studio_app(), host=args.host, port=args.port)
        return
    if not args.reference:  # pragma: no cover
        ap.error("--reference is required unless --studio")
    reg = ModelRegistry(args.runs)  # pragma: no cover
    provider = make_provider(reg)  # pragma: no cover
    uvicorn.run(create_app(provider, args.reference, ref_text=args.ref_text),  # pragma: no cover
                host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
