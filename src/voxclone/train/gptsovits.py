from voxclone.prep.manifest import ClipRecord

def manifest_to_gptsovits(records: list[ClipRecord], speaker: str,
                          language: str = "EN") -> str:
    """GPT-SoVITS training list format: wav_path|speaker|language|text (one per line)."""
    lines = []
    for r in records:
        text = r.text.strip()
        if not text:
            continue
        lines.append(f"{r.audio_path}|{speaker}|{language}|{text}")
    return "\n".join(lines)


import copy
import json
import os
import subprocess
from pathlib import Path

import yaml

from voxclone.prep.manifest import read_manifest
from voxclone.train.base import TrainResult
from voxclone.common.logging import get_logger

log = get_logger("train.gptsovits")

# --- pinned-repo constants (config.py / webui.py, version = "v2Pro") ----------
# All paths are RELATIVE to the GPT-SoVITS repo root; webui.py launches every stage
# from that cwd, so we run the injected runner with cwd=self.root and keep these
# relative (the prep + train scripts resolve them against cwd).
VERSION = "v2Pro"
EXP_ROOT = "logs"                                  # webui.py: exp_root = "logs"
BERT_DIR = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"   # config.py bert_path
CNHUBERT_DIR = "GPT_SoVITS/pretrained_models/chinese-hubert-base"         # config.py cnhubert_path
SV_PATH = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"  # webui.py sv_path
# config.py pretrained_sovits_name["v2Pro"] / pretrained_gpt_name["v2Pro"]
PRETRAINED_S2G = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
PRETRAINED_S1 = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
S2_CONFIG = "GPT_SoVITS/configs/s2v2Pro.json"      # webui.py open1Ba/open1c for v2Pro
S1_CONFIG = "GPT_SoVITS/configs/s1longer-v2.yaml"  # webui.py open1Bb (version != v1)
# config.py SoVITS_weight_version2root / GPT_weight_version2root
SOVITS_WEIGHT_DIR = "SoVITS_weights_v2Pro"
GPT_WEIGHT_DIR = "GPT_weights_v2Pro"

# webui.py hard-caps SoVITS at 25 epochs / GPT at 25; the research doc recommends ~8 / ~15.
DEFAULT_EPOCHS_SOVITS = 8
DEFAULT_EPOCHS_GPT = 15
# webui.py s2 default batch_size is 12 in the UI (config base is 32); pick the webui UI value.
DEFAULT_BATCH_SIZE = 12

# Fallback base configs used ONLY when the real config files are not present under the
# repo root (e.g. in unit tests). When the real files exist we load and template THEM,
# exactly like webui.py. Values mirror GPT_SoVITS/configs/s2v2Pro.json and s1longer-v2.yaml.
_FALLBACK_S2 = {
    "train": {
        "log_interval": 100, "eval_interval": 500, "seed": 1234, "epochs": 100,
        "learning_rate": 0.0001, "betas": [0.8, 0.99], "eps": 1e-09, "batch_size": 32,
        "fp16_run": True, "lr_decay": 0.999875, "segment_size": 20480,
        "init_lr_ratio": 1, "warmup_epochs": 0, "c_mel": 45, "c_kl": 1.0,
        "text_low_lr_rate": 0.4, "grad_ckpt": False,
    },
    "data": {
        "max_wav_value": 32768.0, "sampling_rate": 32000, "filter_length": 2048,
        "hop_length": 640, "win_length": 2048, "n_mel_channels": 128, "mel_fmin": 0.0,
        "mel_fmax": None, "add_blank": True, "n_speakers": 300, "cleaned_text": True,
    },
    "model": {
        "inter_channels": 192, "hidden_channels": 192, "filter_channels": 768,
        "n_heads": 2, "n_layers": 6, "kernel_size": 3, "p_dropout": 0.0,
        "resblock": "1", "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_rates": [10, 8, 2, 2, 2], "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16, 16, 8, 2, 2], "n_layers_q": 3,
        "use_spectral_norm": False, "gin_channels": 1024,
        "semantic_frame_rate": "25hz", "freeze_quantizer": True,
    },
    "s2_ckpt_dir": "logs/s2/big2k1",
    "content_module": "cnhubert",
}
_FALLBACK_S1 = {
    "train": {
        "seed": 1234, "epochs": 20, "batch_size": 8, "save_every_n_epoch": 1,
        "precision": "16-mixed", "gradient_clip": 1.0,
    },
    "optimizer": {
        "lr": 0.01, "lr_init": 0.00001, "lr_end": 0.0001,
        "warmup_steps": 2000, "decay_steps": 40000,
    },
    "data": {"max_eval_sample": 8, "max_sec": 54, "num_workers": 4, "pad_val": 1024},
    "model": {
        "vocab_size": 1025, "phoneme_vocab_size": 732, "embedding_dim": 512,
        "hidden_dim": 512, "head": 16, "linear_units": 2048, "n_layer": 24,
        "dropout": 0, "EOS": 1024, "random_bert": 0,
    },
    "inference": {"top_k": 15},
}


class GPTSoVITSTrainer:
    """REAL GPT-SoVITS v2Pro fine-tune pipeline (comparison-only in this bake-off).

    Faithfully reproduces webui.py's env-driven 4-stage prep + config-file-driven
    s2/s1 training. The subprocess runner is injectable (default subprocess.run); the
    whole flow is GPU-free unit-testable by mocking it, so no line needs pragma-no-cover
    here -- the only real heavy/subprocess work is inside the runner the caller injects.
    """

    def __init__(self, gptsovits_root: str, speaker: str = "target",
                 exp_name: str = "danil", conda_env: str = "gptsovits",
                 runner=subprocess.run):
        self.root = gptsovits_root
        self.speaker = speaker
        self.exp_name = exp_name
        self.conda_env = conda_env
        self.runner = runner

    # --- helpers --------------------------------------------------------------
    def _opt_dir(self) -> str:
        """logs/<exp_name> (webui.py: '%s/%s' % (exp_root, exp_name)). Absolute under root."""
        return str(Path(self.root) / EXP_ROOT / self.exp_name)

    def _conda(self, inner: list) -> list:
        """Wrap a python command to run INSIDE the gptsovits conda env (which has torch +
        the GPT-SoVITS deps + models); bare `python` is the wrong interpreter."""
        return ["conda", "run", "--no-capture-output", "-n", self.conda_env, *inner]

    def _env_with_npp(self, env_extra: dict) -> dict:
        """os.environ + stage env vars + the NVIDIA NPP lib dir on LD_LIBRARY_PATH
        (torchcodec/torchaudio in this env link libnppicc.so.12; the env's NPP path is not
        persisted, so the audio-decoding prep stages need it prepended)."""
        env = os.environ.copy()
        env.update({k: str(v) for k, v in env_extra.items()})
        npp = f"{os.path.expanduser('~')}/miniforge3/envs/{self.conda_env}/lib/python3.11/site-packages/nvidia/npp/lib"
        env["LD_LIBRARY_PATH"] = npp + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        return env

    def _run_stage(self, script_rel: str, env_extra: dict, label: str) -> None:
        """Run one pipeline stage IN the gptsovits env (cwd=root), raising on failure."""
        cmd = self._conda(["python", "-s", script_rel])
        env = self._env_with_npp(env_extra)
        log.info("gptsovits stage %s: %s", label, " ".join(cmd))
        res = self.runner(cmd, env=env, cwd=self.root)
        if getattr(res, "returncode", 0) != 0:
            raise RuntimeError(f"GPT-SoVITS stage failed ({label}): {' '.join(cmd)}")

    def _load_base(self, rel_path: str, fallback: dict) -> dict:
        """Load the real base config from under the repo root, else use the bundled fallback."""
        p = Path(self.root) / rel_path
        if p.exists():
            if p.suffix == ".json":
                return json.loads(p.read_text(encoding="utf-8"))
            return yaml.safe_load(p.read_text(encoding="utf-8"))
        return copy.deepcopy(fallback)

    # --- public ---------------------------------------------------------------
    def train(self, manifest_path: str, out_dir: str, config: dict) -> TrainResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        opt_dir = self._opt_dir()
        Path(opt_dir).mkdir(parents=True, exist_ok=True)

        # (a) write the .list (wav|spk|EN|text) into opt_dir; it is the prep stages' inp_text.
        records = read_manifest(manifest_path)
        list_text = manifest_to_gptsovits(records, speaker=self.speaker)
        list_file = Path(opt_dir) / f"{self.exp_name}.list"
        list_file.write_text(list_text + "\n", encoding="utf-8")
        inp_text = str(list_file)
        # webui derives inp_wav_dir from the .list's absolute paths; the empty string lets the
        # prep scripts use the per-line absolute wav paths directly (matches webui when the
        # slicer output dir is left blank).
        inp_wav_dir = ""

        epochs_sovits = int(config.get("epochs_sovits", DEFAULT_EPOCHS_SOVITS))
        epochs_gpt = int(config.get("epochs_gpt", DEFAULT_EPOCHS_GPT))
        batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))

        # Common env shared by the prep stages (single-GPU: i_part=0, all_parts=1).
        common = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": self.exp_name,
            "opt_dir": opt_dir,
            "version": VERSION,
            "is_half": "True",
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": "0",
        }

        # (b) 4 prep stages IN ORDER (env-driven; webui open1a / open1b / open1c) ----
        # 1-get-text.py: phonemes + BERT features -> logs/<exp>/2-name2text.txt
        self._run_stage(
            "GPT_SoVITS/prepare_datasets/1-get-text.py",
            {**common, "bert_pretrained_dir": BERT_DIR},
            "1-get-text",
        )
        # 2-get-hubert-wav32k.py: HuBERT feats + 32k wav
        self._run_stage(
            "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
            {**common, "cnhubert_base_dir": CNHUBERT_DIR, "sv_path": SV_PATH},
            "2-get-hubert",
        )
        # 2-get-sv.py: v2Pro-MANDATORY speaker-verification embeddings -> 7-sv_cn/
        # (without it v2Pro silently degrades / training asserts in module/data_utils.py).
        self._run_stage(
            "GPT_SoVITS/prepare_datasets/2-get-sv.py",
            {**common, "sv_path": SV_PATH},
            "2-get-sv",
        )
        # 3-get-semantic.py: VQ semantic tokens -> 6-name2semantic.tsv
        self._run_stage(
            "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
            {**common, "pretrained_s2G": PRETRAINED_S2G, "s2config_path": S2_CONFIG},
            "3-get-semantic",
        )

        # (c) s2 (SoVITS) config JSON + s2_train.py --config <tmp_s2.json> -------------
        s2 = self._load_base(S2_CONFIG, _FALLBACK_S2)
        s2["train"]["batch_size"] = batch_size
        s2["train"]["epochs"] = epochs_sovits
        s2["train"]["pretrained_s2G"] = PRETRAINED_S2G
        s2["train"]["pretrained_s2D"] = PRETRAINED_S2G.replace("s2G", "s2D")  # webui.py
        s2["model"]["version"] = VERSION
        s2["data"]["exp_dir"] = s2["s2_ckpt_dir"] = opt_dir
        s2["save_weight_dir"] = SOVITS_WEIGHT_DIR
        s2["name"] = self.exp_name
        s2["version"] = VERSION
        tmp_s2 = out / "tmp_s2.json"
        tmp_s2.write_text(json.dumps(s2), encoding="utf-8")
        self._run_s2(str(tmp_s2))

        # (d) s1 (GPT) config YAML + s1_train.py --config_file <tmp_s1.yaml> -----------
        s1 = self._load_base(S1_CONFIG, _FALLBACK_S1)
        s1["train"]["batch_size"] = batch_size
        s1["train"]["epochs"] = epochs_gpt
        s1["train"]["exp_name"] = self.exp_name
        s1["train"]["half_weights_save_dir"] = GPT_WEIGHT_DIR
        s1["pretrained_s1"] = PRETRAINED_S1
        s1["train_semantic_path"] = f"{opt_dir}/6-name2semantic.tsv"
        s1["train_phoneme_path"] = f"{opt_dir}/2-name2text.txt"
        s1["output_dir"] = f"{opt_dir}/logs_s1_{VERSION}"
        tmp_s1 = out / "tmp_s1.yaml"
        tmp_s1.write_text(yaml.dump(s1, default_flow_style=False), encoding="utf-8")
        self._run_s1(str(tmp_s1))

        # (e) weights land in SoVITS_weights_v2Pro/ + GPT_weights_v2Pro/ under the root.
        sovits_dir = str(Path(self.root) / SOVITS_WEIGHT_DIR)
        return TrainResult(checkpoint_dir=sovits_dir, steps=epochs_sovits + epochs_gpt)

    # --- training entrypoints (config-file driven; NEVER --list/--exp/--epochs) -----
    def _run_s2(self, tmp_config_path: str) -> None:
        cmd = self._conda(["python", "-s", "GPT_SoVITS/s2_train.py", "--config", tmp_config_path])
        env = self._env_with_npp({})
        log.info("gptsovits SoVITS train: %s", " ".join(cmd))
        res = self.runner(cmd, env=env, cwd=self.root)
        if getattr(res, "returncode", 0) != 0:
            raise RuntimeError(f"GPT-SoVITS SoVITS (s2) training failed: {' '.join(cmd)}")

    def _run_s1(self, tmp_config_path: str) -> None:
        # s1_train.py accepts both -c and --config_file; webui.py uses --config_file.
        cmd = self._conda(["python", "-s", "GPT_SoVITS/s1_train.py", "--config_file", tmp_config_path])
        env = self._env_with_npp({"hz": "25hz"})  # webui.py open1Bb: os.environ["hz"] = "25hz"
        log.info("gptsovits GPT train: %s", " ".join(cmd))
        res = self.runner(cmd, env=env, cwd=self.root)
        if getattr(res, "returncode", 0) != 0:
            raise RuntimeError(f"GPT-SoVITS GPT (s1) training failed: {' '.join(cmd)}")
