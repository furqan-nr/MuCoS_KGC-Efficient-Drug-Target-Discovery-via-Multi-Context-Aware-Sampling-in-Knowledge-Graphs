import torch
import pandas as pd
import warnings

from utils import load_triplets
from model import get_tokenizer_and_model_classes
from train import train_and_evaluate

# ====================== CONFIG ======================
train_file_path = 'D:/GPKG-predict/kegg50k/train.txt'
valid_file_path = 'D:/GPKG-predict/kegg50k/valid_o.txt'
test_file_path  = 'D:/GPKG-predict/kegg50k/test_o.txt'
model_save_path = 'D:/GPKG-predict/Specific_r_pred_in_kegg50k'

NUM_EPOCHS = 50
MODEL_NAME = "distilbert-base-uncased"
# ===================================================

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

if __name__ == '__main__':
    # Load benchmark triplets without domain-specific filtering.
    train_triplets = load_triplets(train_file_path)
    valid_triplets = load_triplets(valid_file_path)
    test_triplets  = load_triplets(test_file_path)
    
    print("Training triplets shape:", train_triplets.shape)
    print("Validation triplets shape:", valid_triplets.shape)
    print("Test triplets shape:", test_triplets.shape)
    
    # Relation mapping from the training split.
    relations = train_triplets['relation'].unique().tolist()
    relation_to_idx = {rel: idx for idx, rel in enumerate(relations)}
    
    # Training-only graph context avoids leakage into validation/test.
    all_triplets = train_triplets
    
    # Get tokenizer and model
    tokenizer_class, model_class = get_tokenizer_and_model_classes(MODEL_NAME)
    
    # Start training
    train_and_evaluate(
        model_name=MODEL_NAME,
        tokenizer_class=tokenizer_class,
        model_class=model_class,
        save_path=model_save_path,
        num_epochs=NUM_EPOCHS,
        train_triplets=train_triplets,
        valid_triplets=valid_triplets,
        test_triplets=test_triplets,
        relation_to_idx=relation_to_idx,
        all_triplets=all_triplets,
        device=device
    )