import json
import os
import random

import torch
import numpy as np
from transformers import DataCollatorWithPadding


class CollateWithPadding:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        labels = []
        meta = []
        for _, label, metadata in batch:
            labels.append(label.item() if isinstance(label, torch.Tensor) else int(label))
            meta.append(metadata)

        inputs = self.data_collator(inputs)
        labels = torch.tensor(labels, dtype=torch.long)
        return inputs, labels, meta

def save_test_results(epoch, test_results, save_path):
    """Save test results to a file."""
    # Ensure the save path directory exists
    os.makedirs(save_path, exist_ok=True)

    # Path to the result file
    file_path = os.path.join(save_path, "test_results.txt")

    # Prepare the results for the current epoch as a single line
    epoch_results = (f"{test_results['MRR']:.4f}\t"
                     f"{test_results['Hits@1']:.4f}\t"
                     f"{test_results['Hits@3']:.4f}\t"
                     f"{test_results['Hits@5']:.4f}\t"
                     f"{test_results['Hits@10']:.4f}\n")

    # Check if the file already exists
    if not os.path.exists(file_path):
        # If file doesn't exist, write the header first
        with open(file_path, 'w') as file:
            file.write("MRR\tHit@1\tHit@3\tHit@5\tHit@10\n")

    # Append the epoch results to the file
    with open(file_path, 'a') as file:
        file.write(epoch_results)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def capture_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and "torch_cuda_random_state_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_random_state_all"])


def save_checkpoint(model, optimizer, checkpoint_path, **metadata):
    """Save a training checkpoint with optional metadata for exact resume."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **metadata,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load a training checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
    return None


def build_collate_fn(tokenizer, pad_to_multiple_of=8):
    return CollateWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of)