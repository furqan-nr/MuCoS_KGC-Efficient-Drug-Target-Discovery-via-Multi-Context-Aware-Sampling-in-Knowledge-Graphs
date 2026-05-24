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

# Context budget constants
MAX_HC = 15
MAX_RC = 5
MAX_TOTAL_CONTEXT = 20
MAX_RC_IF_HC_SHORT = 10
MAX_SAME_RELATION_IN_HC = 5

# Reproducibility
SEED = 42