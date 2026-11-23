import random
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "notebooks" / "data"
CHECKPOINT_PATH = BASE_DIR / "models" / "vit_emnist.pt"

SEED = 42
BATCH_SIZE = 256
N_CALIB = 2000
N_ID_EVAL = 2000

N_KMNIST = 200
N_FASHION = 150
N_SCRIBBLES = 150

N_PASSES = 20
UNC_THRESHOLD = 0.004
OOD_PERCENTILE = 95.0


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_emnist_char(label: int) -> str:
    if label < 10:
        return str(label)
    if label < 36:
        return chr(ord("A") + (label - 10))
    return chr(ord("a") + (label - 36))