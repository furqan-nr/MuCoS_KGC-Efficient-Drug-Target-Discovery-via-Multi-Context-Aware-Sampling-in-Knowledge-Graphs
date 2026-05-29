import os
import torch

# Device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Data paths (override with environment variables if needed)
DATA_DIR = os.getenv("MUCOS_DATA_DIR", "data")
train_file_path = os.getenv("MUCOS_TRAIN_PATH", os.path.join(DATA_DIR, "train.txt"))
valid_file_path = os.getenv("MUCOS_VALID_PATH", os.path.join(DATA_DIR, "valid.txt"))
test_file_path = os.getenv("MUCOS_TEST_PATH", os.path.join(DATA_DIR, "test.txt"))

# Output paths
PROCESSED_DIR = os.getenv("MUCOS_PROCESSED_DIR", "processed")
OUTPUT_DIR = os.getenv("MUCOS_OUTPUT_DIR", "outputs")

# Model hyperparameters
MODEL_NAME = "distilbert-base-uncased"  # or "bert-base-uncased", "roberta-base"
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_LENGTH = 128

# Preprocessing/cache controls
SAVE_TOKENIZED_CACHE = os.getenv("MUCOS_SAVE_TOKENIZED_CACHE", "1") == "1"
REQUIRE_TOKENIZED_CACHE = os.getenv("MUCOS_REQUIRE_TOKENIZED_CACHE", "0") == "1"

# Runtime training controls
USE_AMP = os.getenv("MUCOS_USE_AMP", "1") == "1"
PAD_TO_MULTIPLE_OF = int(os.getenv("MUCOS_PAD_TO_MULTIPLE_OF", "8"))

# Optional relation-tail prior reranking
RELATION_PRIOR_ENABLED = os.getenv("MUCOS_RELATION_PRIOR_ENABLED", "0") == "1"
RELATION_PRIOR_ALPHAS = os.getenv("MUCOS_RELATION_PRIOR_ALPHAS", "0.0,0.05,0.1,0.2")

# Optional type-constrained evaluation (relation -> allowed tail set)
TYPE_CONSTRAINT_ENABLED = os.getenv("MUCOS_TYPE_CONSTRAINT_ENABLED", "0") == "1"
TYPE_CONSTRAINT_MIN_K = int(os.getenv("MUCOS_TYPE_CONSTRAINT_MIN_K", "0"))
TYPE_CONSTRAINT_FALLBACK = os.getenv("MUCOS_TYPE_CONSTRAINT_FALLBACK", "1") == "1"

# Context budget constants
MAX_HC = 15
MAX_RC = 5
MAX_TOTAL_CONTEXT = 20
MAX_RC_IF_HC_SHORT = 10
MAX_SAME_RELATION_IN_HC = 5

# Input format control (default keeps original order)
# Options: default, prioritize_relation, auto
CONTEXT_ORDER = os.getenv("MUCOS_CONTEXT_ORDER", "default")

# Reproducibility
SEED = 42

# Laptop-friendly opt-in controls
RESUME_TRAINING = os.getenv("MUCOS_RESUME_TRAINING", "1") == "1"
# Save intermediate checkpoints every N optimizer steps. Set to 0 to disable.
# Default to a conservative small value so laptop users can interrupt safely.
CHECKPOINT_EVERY_STEPS = int(os.getenv("MUCOS_CHECKPOINT_EVERY_STEPS", "100"))
MAX_TRAIN_SECONDS = float(os.getenv("MUCOS_MAX_TRAIN_SECONDS", "0"))

# Runtime logging controls
LOG_EVERY_STEPS = int(os.getenv("MUCOS_LOG_EVERY_STEPS", "100"))