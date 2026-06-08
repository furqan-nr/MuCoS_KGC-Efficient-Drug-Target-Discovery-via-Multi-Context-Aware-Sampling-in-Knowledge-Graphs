import os
import numpy as np
import torch


def _rank_batch(logits, labels):
    scores = torch.softmax(logits, dim=-1)
    sorted_indices = torch.argsort(scores, dim=-1, descending=True)
    ranks = []
    for row_indices, label in zip(sorted_indices, labels):
        rank = (row_indices == label).nonzero(as_tuple=False).item() + 1
        ranks.append(rank)
    return ranks



# ==================== EVALUATION FUNCTION ====================
# This function evaluates a trained relation prediction model on a test dataset.
# - Handles both standard and DDP-wrapped models.
# - Computes rankings of the true relation among all possible relations using softmax probabilities.
# - Calculates standard knowledge graph metrics:
#     - MR  (Mean Rank)
#     - MRR (Mean Reciprocal Rank)
#     - Hits@K for K = 1, 3, 5, 10
# - Returns a dictionary containing all metrics for easy logging or saving.

def evaluate_relation_model(model, dataloader, device, relation_to_idx, test_triplets,
                            tokenizer, entity_incoming_neighbors, max_length=128):
    """Evaluate model on test data for relation prediction."""
    
    # Unwrap DDP/DataParallel
    if hasattr(model, 'module'):
        eval_model = model.module
    else:
        eval_model = model
    
    eval_model.eval()
    rankings = []

    # Prepare list of all relations for ranking
    all_relations = list(relation_to_idx.keys())
    num_relations = len(all_relations)
    
    with torch.no_grad():
        for batch in dataloader:
            # dataloader may yield (inputs, labels) or (inputs, labels, meta)
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels, _meta = batch

            inputs = {key: val.to(device, non_blocking=True) for key, val in inputs.items()}
            labels = labels.to(device, non_blocking=True)

            outputs = eval_model(**inputs)
            logits = outputs.logits
            rankings.extend(_rank_batch(logits, labels))

    rankings = np.array(rankings)
    
    # Calculate metrics
    mr = np.mean(rankings)                     # Mean Rank
    mrr = np.mean(1.0 / rankings)              # Mean Reciprocal Rank
    hits_at_1 = np.mean(rankings <= 1)
    hits_at_3 = np.mean(rankings <= 3)
    hits_at_5 = np.mean(rankings <= 5)
    hits_at_10 = np.mean(rankings <= 10)
    
    return {
        'MR': mr,
        'MRR': mrr,
        'Hits@1': hits_at_1,
        'Hits@3': hits_at_3,
        'Hits@5': hits_at_5,
        'Hits@10': hits_at_10
    }




# ==================== SAVE TEST RESULTS ====================
# Saves evaluation metrics to a text file.
# - Creates file if it doesn't exist, adds a header.
# - Appends results for each epoch.
# - Helps maintain logs for multiple epochs and datasets.

def save_test_results(epoch, test_results, save_path, task='relation'):
    """Save test results to file including MR."""
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{task}_test_results.txt")
    
    epoch_results = (f"{test_results['MR']:.2f}\t"
                     f"{test_results['MRR']:.4f}\t"
                     f"{test_results['Hits@1']:.4f}\t"
                     f"{test_results['Hits@3']:.4f}\t"
                     f"{test_results['Hits@5']:.4f}\t"
                     f"{test_results['Hits@10']:.4f}\n")
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("Epoch\tMR\tMRR\tHits@1\tHits@3\tHits@5\tHits@10\n")
    
    with open(file_path, 'a') as file:
        file.write(f"{epoch + 1}\t{epoch_results.strip()}\n")



# ==================== CHECKPOINT SAVE / LOAD ====================
# These functions save and load model checkpoints during training.
# - Includes model state, optimizer state, and current epoch.
# - Works for both standard and DDP-wrapped models.
# - Allows resuming training from the last checkpoint without losing progress.

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    else:
        return 0


def hard_negative_hinge_loss(logits: torch.Tensor, labels: torch.Tensor,
                             top_k: int = 1, margin: float = 0.5, weight: float = 1.0) -> torch.Tensor:
    """Compute hinge loss on the hardest incorrect label(s) per sample.

    logits: (batch, num_classes), labels: (batch,)
    Returns scalar loss.
    """
    if top_k <= 0:
        return torch.tensor(0.0, device=logits.device)

    # copy logits and mask the positive class
    logits_clone = logits.clone()
    logits_clone[torch.arange(logits.size(0)), labels] = -1e9

    pos_scores = logits[torch.arange(logits.size(0)), labels]
    topk_vals, _ = torch.topk(logits_clone, top_k, dim=1)
    # margin loss: max(0, margin - (pos - neg))
    diffs = margin - (pos_scores.unsqueeze(1) - topk_vals)
    losses = torch.clamp(diffs, min=0.0)
    return weight * losses.mean()