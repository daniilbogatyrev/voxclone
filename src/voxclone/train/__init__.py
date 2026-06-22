from voxclone.train.gptsovits import GPTSoVITSTrainer
from voxclone.train.xtts import XTTSTrainer
from voxclone.train.f5 import F5Trainer

# Engine name -> TrainAdapter class. The key is the load-bearing engine name
# (== --engine value == --out basename == registry model key == SYNTH_ENGINES key).
# Chatterbox is intentionally absent: it is zero-shot only and is NEVER fine-tuned.
TRAIN_ENGINES = {
    "xtts": XTTSTrainer,
    "f5": F5Trainer,
    "gptsovits": GPTSoVITSTrainer,
}
