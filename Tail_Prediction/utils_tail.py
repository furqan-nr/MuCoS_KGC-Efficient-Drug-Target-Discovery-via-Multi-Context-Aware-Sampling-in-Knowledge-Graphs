import json
import os

import torch

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

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Save a training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load a training checkpoint if it exists."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    else:
        return 0  # Start from scratch