import json
import os
import random
import time

import numpy as np
import torch

try:
    from transformers import set_seed as hf_set_seed
except ImportError:
    hf_set_seed = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if hf_set_seed:
        hf_set_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def now_seconds():
    return time.perf_counter()
