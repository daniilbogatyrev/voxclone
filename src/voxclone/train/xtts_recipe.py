"""XTTS-v2 fine-tune driver (GPU-only, runs INSIDE the ``xtts`` conda env).

This is the thin recipe ``voxclone.train.xtts.XTTSTrainer`` shells out to:

    conda run -n xtts python -m voxclone.train.xtts_recipe \\
        --model_dir <xtts_v2_model> --train_csv ... --eval_csv ... --out ... \\
        --epochs ... --batch_size ... --grad_accum ... --lr ... \\
        --mixed_precision True --max_wav_length 255995 --max_text_length 200

It drives the coqui ``GPTTrainer`` recipe over the two coqui-format CSVs that
``manifest_to_xtts`` writes (formatter ``"coqui"`` -- NOT ``ljspeech``). All heavy work
(torch / TTS / trainer) is GPU-only and lives inside ``main`` so importing this module is
side-effect-free; the GPU-free test suite never imports it (``XTTSTrainer`` mocks the
subprocess ``runner``). The whole live path is therefore ``# pragma: no cover``.

Requirements in ``--model_dir`` (the HF ``coqui/XTTS-v2`` snapshot already contains all of
these): ``config.json``, ``model.pth``, ``vocab.json``, ``dvae.pth``, ``mel_stats.pth``.
Needs torch>=2.6's ``add_safe_globals`` (idiap coqui-tts 0.27.5, transformers<=4.57.6).
"""
import argparse
import os


def _build_parser() -> argparse.ArgumentParser:
    """Argument contract -- MUST match the command XTTSTrainer.train constructs."""
    p = argparse.ArgumentParser(description="Fine-tune XTTS-v2 via the coqui GPTTrainer recipe.")
    p.add_argument("--model_dir", required=True, help="base XTTS-v2 model dir (config/model/vocab/dvae/mel_stats)")
    p.add_argument("--train_csv", required=True, help="coqui-format metadata_train.csv (audio_file|text|speaker_name)")
    p.add_argument("--eval_csv", required=True, help="coqui-format metadata_eval.csv (in-training loss split)")
    p.add_argument("--out", required=True, help="output dir for checkpoints/logs (== runs/xtts)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--mixed_precision", type=_str2bool, default=True)
    p.add_argument("--max_wav_length", type=int, default=255995, help="samples; XTTS DROPS longer clips")
    p.add_argument("--max_text_length", type=int, default=200, help="chars; XTTS DROPS longer texts")
    return p


def _str2bool(v) -> bool:
    """Parse the str(bool) the trainer passes (\"True\"/\"False\")."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def main(argv=None) -> None:  # pragma: no cover (GPU-only)
    args = _build_parser().parse_args(argv)

    import torch
    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import (
        GPTArgs,
        GPTTrainer,
        GPTTrainerConfig,
    )
    from TTS.tts.configs.xtts_config import XttsConfig
    # XttsAudioConfig lives in the model module (NOT gpt_trainer) in idiap coqui-tts 0.27.5.
    from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig

    # torch>=2.6 weights_only: allow the XTTS config/arg dataclasses during checkpoint load.
    torch.serialization.add_safe_globals(
        [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig]
    )

    os.makedirs(args.out, exist_ok=True)
    xtts_checkpoint = os.path.join(args.model_dir, "model.pth")
    tokenizer_file = os.path.join(args.model_dir, "vocab.json")
    dvae_checkpoint = os.path.join(args.model_dir, "dvae.pth")
    mel_norm_file = os.path.join(args.model_dir, "mel_stats.pth")
    for path in (xtts_checkpoint, tokenizer_file, dvae_checkpoint, mel_norm_file):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"XTTS fine-tune needs {path}; the coqui/XTTS-v2 HF snapshot ships it -- "
                f"ensure --model_dir points at a full snapshot (dvae.pth + mel_stats.pth included)."
            )

    # Coqui formatter computes each clip's path RELATIVE TO root_path (add_extra_keys), so
    # root_path MUST be a common ancestor of the (absolute) audio files -- NOT the CSV dir.
    audio_dirs = []
    with open(args.train_csv, encoding="utf-8") as _f:
        for _line in _f.read().splitlines()[1:]:          # skip the header row
            if _line.strip():
                audio_dirs.append(os.path.dirname(_line.split("|")[0]))
    root_path = os.path.commonpath(audio_dirs) if audio_dirs else os.path.sep

    dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="danil",
        path=root_path,
        meta_file_train=os.path.abspath(args.train_csv),
        meta_file_val=os.path.abspath(args.eval_csv),
        language="en",
    )

    model_args = GPTArgs(
        max_conditioning_length=132300,   # 6 s @ 22.05 kHz
        min_conditioning_length=66150,    # 3 s
        max_wav_length=args.max_wav_length,
        max_text_length=args.max_text_length,
        mel_norm_file=mel_norm_file,
        dvae_checkpoint=dvae_checkpoint,
        xtts_checkpoint=xtts_checkpoint,
        tokenizer_file=tokenizer_file,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000
    )
    config = GPTTrainerConfig(
        output_path=args.out,
        model_args=model_args,
        audio=audio_config,
        run_name="xtts_finetune_danil",
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        batch_group_size=48,
        num_loader_workers=8,
        print_step=50,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5, "last_epoch": -1},
        mixed_precision=args.mixed_precision,
    )

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        [dataset], eval_split=True, eval_split_max_size=256, eval_split_size=0.01
    )
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=args.grad_accum,
        ),
        config,
        output_path=args.out,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":  # pragma: no cover
    main()
