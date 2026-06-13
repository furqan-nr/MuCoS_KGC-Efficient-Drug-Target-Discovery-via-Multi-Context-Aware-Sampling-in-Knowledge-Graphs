"""Configuration for negative-sample-free relation prediction on general KG benchmarks.

This configuration is no longer biomedical-specific. It targets standard KGC
relation-prediction benchmarks such as FB15k-237, WN18RR, CoDEx-S/M, and
YAGO3-10. FB15K/WN18 are intentionally not included in the main list because
older inverse-relation leakage makes them poor primary benchmarks.
"""

import os
import torch


# ==================== DATASETS ====================
# Edit paths for your local environment. Optional label files are TSV files:
#   raw_id<TAB>readable label
# Entity/relation label files are strongly recommended for FB15k-237, CoDEx,
# and YAGO because raw IDs often have weak lexical meaning for BERT.
DATASETS = [
    {
        "name": "FB15k-237",
        "TRAIN_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15k-237/train.txt",
        "VALID_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15k-237/valid.txt",
        "TEST_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15k-237/test.txt",
        "MODEL_SAVE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15k-237/apc-mucos-distilbert",
        "ENTITY_TYPE_FILE": None,
        "ENTITY_LABEL_FILE": None,
        "RELATION_LABEL_FILE": None,
    },
    {
        "name": "WN18RR",
        "TRAIN_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18RR/train.txt",
        "VALID_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18RR/valid.txt",
        "TEST_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18RR/test.txt",
        "MODEL_SAVE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18RR/apc-mucos-distilbert",
        "ENTITY_TYPE_FILE": None,
        "ENTITY_LABEL_FILE": None,
        "RELATION_LABEL_FILE": None,
    },
    {
        "name": "CoDEx-S",
        "TRAIN_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/CoDEx-S/train.txt",
        "VALID_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/CoDEx-S/valid.txt",
        "TEST_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/CoDEx-S/test.txt",
        "MODEL_SAVE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/CoDEx-S/apc-mucos-distilbert",
        "ENTITY_TYPE_FILE": None,
        "ENTITY_LABEL_FILE": None,
        "RELATION_LABEL_FILE": None,
    },
    {
        "name": "CoDEx-M",
        "TRAIN_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/CoDEx-M/train.txt",
        "VALID_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/CoDEx-M/valid.txt",
        "TEST_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/CoDEx-M/test.txt",
        "MODEL_SAVE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/CoDEx-M/apc-mucos-distilbert",
        "ENTITY_TYPE_FILE": None,
        "ENTITY_LABEL_FILE": None,
        "RELATION_LABEL_FILE": None,
    },
    {
        "name": "YAGO3-10",
        "TRAIN_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/YAGO3-10/train.txt",
        "VALID_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/YAGO3-10/valid.txt",
        "TEST_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/YAGO3-10/test.txt",
        "MODEL_SAVE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/YAGO3-10/apc-mucos-distilbert",
        "ENTITY_TYPE_FILE": None,
        "ENTITY_LABEL_FILE": None,
        "RELATION_LABEL_FILE": None,
    },
]


# ==================== REPRODUCIBILITY ====================
SEED = 42
DETERMINISTIC = False


# ==================== DEVICE & GPU SETUP ====================
if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    DEVICE = torch.device("cuda")
else:
    NUM_GPUS = 0
    DEVICE = torch.device("cpu")

MASTER_PORT = os.environ.get("MASTER_PORT", "29500")
DDP_BACKEND = "nccl"


# ==================== MODEL CONFIGURATION ====================
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 64
PER_GPU_BATCH_SIZE = max(1, BATCH_SIZE // max(NUM_GPUS, 1))
LEARNING_RATE = 5e-5
NUM_EPOCHS = 20
GRADIENT_ACCUMULATION_STEPS = 2
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0
SAVE_EVERY_EPOCH = False

# Checkpoint selection. "mrr" is the classic KGC choice; "mrr_macro_f1" is
# better when rare relation performance matters.
CHECKPOINT_METRIC = "mrr_macro_f1"  # mrr | macro_f1 | mrr_macro_f1
CHECKPOINT_MRR_WEIGHT = 0.7
CHECKPOINT_MACRO_F1_WEIGHT = 0.3


# ==================== DATALOADER / TOKENIZATION ====================
NUM_WORKERS = 4
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# Dynamic padding is preferred for standard KG benchmarks: it avoids padding all
# examples to MAX_LENGTH and directly improves memory/time without changing the
# model. Set PRETOKENIZE=True only for small smoke tests or when fixed tensors
# are needed.
PRETOKENIZE = False
DYNAMIC_PADDING = True


# ==================== STRUCTURED INPUT TOKENS ====================
SPECIAL_TOKENS = [
    "[HEAD]", "[TAIL]",
    "[HEAD_TYPE]", "[TAIL_TYPE]", "[HEAD_SCHEMA]", "[TAIL_SCHEMA]",
    "[HEAD_CTX]", "[TAIL_CTX]", "[PAIR_CTX]",
    "[H_IN]", "[H_OUT]", "[T_IN]", "[T_OUT]",
    "[PATH]", "[TYPE]", "[SCHEMA]", "[2HOP]",
]


# ==================== TEXT NORMALIZATION ====================
# Generic KG benchmarks often contain IDs/URIs rather than natural labels. Keep
# raw IDs internally but feed normalized text to the encoder.
NORMALIZE_ENTITY_TEXT = True
NORMALIZE_RELATION_TEXT = True
USE_LABEL_MAPS = True


# ==================== CONTEXT SAMPLING ====================
CONTEXT_MODE = "adaptive"  # fixed | adaptive
MAX_DEGREE = 30
MIN_CONTEXT = 5
BASE_CONTEXT = 15
MAX_CONTEXT = 30

# Token budgets are enforced before tokenizer truncation.
MAX_HEAD_CONTEXT_TOKENS = 40
MAX_TAIL_CONTEXT_TOKENS = 40
MAX_PAIR_CONTEXT_TOKENS = 24
MAX_PATH_CONTEXT_TOKENS = 20

# Relevance-aware sampling weights. Density stays central, but relation rarity
# and redundancy help on long-tail relation benchmarks.
W_DEGREE = 1.0
W_REL_RARITY = 0.35
W_REDUNDANCY = 0.15

# Compress repeated context relation items, e.g. "born_in: A | B | C".
GROUP_CONTEXT_BY_RELATION = True
MAX_NEIGHBORS_PER_RELATION = 3

# Optional second-hop local expansion. Keep off for baseline.
USE_SECOND_HOP_CONTEXT = False
SECOND_HOP_BUDGET = 5


# ==================== PAIR CONTEXT AND PATH MOTIFS ====================
# Pair context is the main general-KG upgrade: relation prediction is about the
# pair (h, ?, t), so shared neighbors and bridge motifs are highly informative.
USE_PAIR_CONTEXT = True
MAX_COMMON_NEIGHBORS = 5
MAX_PAIR_MOTIFS = 5

# Path context is optional. The default mode is compressed relation motifs rather
# than full entity paths to control token cost.
USE_PATH_CONTEXT = False
PATH_CONTEXT_MODE = "motif"  # motif | full
MAX_PATHS = 3
MAX_PATH_LEN = 2


# ==================== ENTITY TYPES / SCHEMA PRIORS ====================
# For general datasets, explicit type files are often missing. Keep entity type
# tokens off by default and use induced schema signatures instead.
USE_ENTITY_TYPES = False
DEFAULT_ENTITY_TYPE = "UNK"
USE_TYPE_RELATION_MASK = False
APPLY_TYPE_MASK_DURING_TRAINING = False

USE_SCHEMA_TOKENS = True
USE_SCHEMA_PRIOR = True
SCHEMA_MODE = "relation_signature"  # relation_signature | degree
SCHEMA_TOP_RELATIONS = 3
SCHEMA_PRIOR_WEIGHT = 0.10
SCHEMA_PRIOR_SMOOTHING = 1.0


# ==================== LOSS / LONG-TAIL RELATIONS ====================
# No negative triples are generated. These are positive-label classification
# strategies only.
LOSS_TYPE = "ce"  # ce | weighted_ce | deferred_weighted_ce | focal
REWEIGHT_START_EPOCH = 3
LABEL_SMOOTHING = 0.0
FOCAL_GAMMA = 2.0
CLASS_WEIGHT_SMOOTHING = 1.2

# Positive-only relation-balanced sampling. In DDP, the implementation falls
# back to DistributedSampler; use this in single-GPU/CPU ablations first.
TRAIN_SAMPLER = "random"  # random | relation_balanced
RELATION_SAMPLER_POWER = 0.5


# ==================== EVALUATION ====================
# Raw relation ranking ranks the gold relation among all relation labels.
# Filtered relation ranking masks other known true relations for the same
# (head, tail) pair, which is important when multiple relations can connect the
# same entity pair.
FILTERED_RELATION_EVAL = True
RARE_RELATION_THRESHOLD = 10
MEDIUM_RELATION_THRESHOLD = 100


# ==================== LOGGING / OUTPUT ====================
METRICS_JSON = "metrics.jsonl"
CONFIG_JSON = "run_config.json"
GRAPH_CACHE_PREFIX = "graph_stats_train_only"
GRAPH_CACHE_VERSION = "v3_general_kg_pair_schema_context"


def print_config_summary():
    """Print a compact runtime summary."""
    print(f"Using Device: {DEVICE}")
    print(f"Number of GPUs: {NUM_GPUS}")
    print(f"Global Batch Size: {BATCH_SIZE}")
    print(f"Per GPU Batch Size: {PER_GPU_BATCH_SIZE}")
    print(f"DDP Master Port: {MASTER_PORT}")
    print(f"Using Model: {MODEL_NAME}")
    print(f"Context Mode: {CONTEXT_MODE}")
    print(f"Dynamic Padding: {DYNAMIC_PADDING}; Pretokenize: {PRETOKENIZE}")
    print(f"Pair Context: {USE_PAIR_CONTEXT}; Schema Prior: {USE_SCHEMA_PRIOR}")
    print(f"Found {len(DATASETS)} main datasets → will train sequentially\n")
