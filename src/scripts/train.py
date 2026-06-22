"""Fine-tune one engine, dispatched through voxclone.train.TRAIN_ENGINES.

`--engine {xtts,f5,gptsovits}` selects the TrainAdapter class; each is constructed
with only its own roots/conda-env args and then `.train(manifest, out, config)` is
called with the YAML's `train:` sub-mapping. The naming contract: `--out` defaults to
`runs/<engine>`, whose basename == the engine key == the registry key == the served
model name. Chatterbox is intentionally NOT trainable (zero-shot only; not in the map).
"""
import argparse

import yaml

from voxclone.train import TRAIN_ENGINES


def _build_trainer(engine: str, args):
    """Construct the selected engine's trainer with ONLY its relevant ctor args.

    Roots are conditional: gptsovits needs --gptsovits-root; xtts optionally takes
    --xtts-root / --xtts-conda-env; f5 optionally takes --f5-conda-env. Unset optional
    args fall through to each trainer's own defaults.
    """
    cls = TRAIN_ENGINES[engine]
    if engine == "gptsovits":
        return cls(gptsovits_root=args.gptsovits_root)
    if engine == "xtts":
        kwargs = {}
        if args.xtts_root is not None:
            kwargs["xtts_root"] = args.xtts_root
        if args.xtts_conda_env is not None:
            kwargs["conda_env"] = args.xtts_conda_env
        return cls(**kwargs)
    if engine == "f5":
        kwargs = {}
        if args.f5_conda_env is not None:
            kwargs["conda_env"] = args.f5_conda_env
        return cls(**kwargs)
    return cls()  # pragma: no cover - choices restrict engine to the branches above


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="fine-tune a model")
    ap.add_argument("--engine", required=True, choices=sorted(TRAIN_ENGINES),
                    help="which engine to fine-tune (selects the TrainAdapter)")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out", default=None,
                    help="checkpoint dir (default runs/<engine>; basename == engine key)")
    ap.add_argument("--config", required=True)
    # Conditional roots / conda envs (validated per --engine below).
    ap.add_argument("--gptsovits-root", help="GPT-SoVITS repo root (required for --engine gptsovits)")
    ap.add_argument("--xtts-root", help="XTTS-v2 model dir (optional; for --engine xtts)")
    ap.add_argument("--xtts-conda-env", help="conda env XTTS trains in (optional; for --engine xtts)")
    ap.add_argument("--f5-conda-env", help="conda env F5-TTS trains in (optional; for --engine f5)")
    args = ap.parse_args(argv)

    if args.engine == "gptsovits" and not args.gptsovits_root:
        ap.error("--gptsovits-root is required for --engine gptsovits")

    out = args.out if args.out is not None else f"runs/{args.engine}"

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    train_cfg = cfg.get("train", {})

    trainer = _build_trainer(args.engine, args)
    result = trainer.train(args.manifest, out, train_cfg)
    print(f"trained {args.engine} -> {result.checkpoint_dir} ({result.steps} steps)")


if __name__ == "__main__":  # pragma: no cover
    main()
