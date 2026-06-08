import os
import torch


# ==================== ALL DATASETS (run sequentially) ====================
# This list contains multiple Knowledge Graph datasets.
# Each dataset has:
#   - name: dataset identifier
#   - TRAIN/VALID/TEST paths: where the triplet files are stored
#   - MODEL_SAVE_PATH: where the trained model will be saved for that dataset
# The code will loop through this list and train the model on each dataset one by one.


DATASETS = [
    
    {
        "name": "FB15k-237",
        "TRAIN_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15k-237/train.txt",
        "VALID_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15k-237/valid.txt",
        "TEST_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15k-237/test.txt",
        "MODEL_SAVE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15k-237/distilbert-base-uncased"
    },
    {
        "name": "FB15K",
        "TRAIN_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15K/train.txt",
        "VALID_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15K/valid.txt",
        "TEST_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15K/test.txt",
        "MODEL_SAVE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/FB15K/distilbert-base-uncased"
    },
    {
        "name": "WN18RR",
        "TRAIN_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18RR/train.txt",
        "VALID_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18RR/valid.txt",
        "TEST_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18RR/test.txt",
        "MODEL_SAVE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18RR/distilbert-base-uncased"
    },
    {
        "name": "WN18",
        "TRAIN_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18/train.txt",
        "VALID_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18/valid.txt",
        "TEST_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18/test.txt",
        "MODEL_SAVE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/WN18/distilbert-base-uncased"
    },
    {
        "name": "YAGO3-10",
        "TRAIN_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/YAGO3-10/train.txt",
        "VALID_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/YAGO3-10/valid.txt",
        "TEST_FILE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/YAGO3-10/test.txt",
        "MODEL_SAVE_PATH": "/home/user/23h1710_KGC/MuCos-KGC/data/YAGO3-10/distilbert-base-uncased"
    }
]




# ==================== DEVICE & GPU SETUP ====================
# This block checks whether a GPU is available.
# If yes → use GPU(s) for faster training
# If not → use CPU (slower but still works)

if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    DEVICE = torch.device('cuda')
    print(f"Using {NUM_GPUS} GPU(s):")
    for i in range(NUM_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    NUM_GPUS = 0
    DEVICE = torch.device('cpu')



# ==================== MODEL CONFIGURATION ====================
# These are training hyperparameters that control how the model learnsn

BATCH_SIZE = 64
PER_GPU_BATCH_SIZE = BATCH_SIZE // NUM_GPUS if NUM_GPUS > 0 else BATCH_SIZE

MAX_LENGTH = 128
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
MAX_DEGREE = 30
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0

# DDP Communication Port
MASTER_PORT = "29500"


# ==================== HARD NEGATIVE MINING (OPTIONAL) ====================
# Lightweight hard-negative hinge loss to emphasize top incorrect labels.
# These settings are conservative defaults—tune per dataset for best results.
HARD_NEGATIVE_MINING_ENABLED = True
HNM_POOL_SIZE = 100  # not used in this simple hinge variant, reserved for future
HNM_TOP_K = 1
HNM_MARGIN = 0.5
HNM_WEIGHT = 1.0
HNM_ENTITY_NEG_K = 1



# ==================== MODEL SELECTION ====================
MODEL_NAME = "distilbert-base-uncased"
# Pretrained language model used for encoding text (lighter and faster than BERT)

# ==================== LOGGING INFO ====================
# Print all important settings before training starts

print(f"Using Device: {DEVICE}")
print(f"Number of GPUs: {NUM_GPUS}")
print(f"Global Batch Size: {BATCH_SIZE}")
print(f"Per GPU Batch Size: {PER_GPU_BATCH_SIZE}")
print(f"DDP Master Port: {MASTER_PORT}")
print(f"Using Model: {MODEL_NAME}")
print(f"Found {len(DATASETS)} datasets → will train sequentially\n")