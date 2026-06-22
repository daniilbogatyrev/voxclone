import argparse
from pathlib import Path
from voxclone.common.config import load_config
from voxclone.common.seed import set_seed
from voxclone.prep.segment import load_vad
from voxclone.prep.transcribe import load_transcriber
from voxclone.prep.pipeline import run_prep

def main() -> None:
    ap = argparse.ArgumentParser(description="voxclone data prep pipeline")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--transcripts", default="data/transcripts.csv")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    run_prep(
        raw_dir=args.raw, out_dir=args.out, config=cfg.prep,
        vad=load_vad(),
        transcriber=load_transcriber(cfg.prep.whisper_model, cfg.prep.whisper_device),
        manifest_path=args.manifest, transcript_csv=args.transcripts,
    )

if __name__ == "__main__":
    main()
