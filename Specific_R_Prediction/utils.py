import os
import pandas as pd
from collections import defaultdict
import torch


def _rank_batch(logits, labels):
    scores = torch.softmax(logits, dim=-1)
    sorted_indices = torch.argsort(scores, dim=-1, descending=True)
    ranks = []
    for row_indices, label in zip(sorted_indices, labels):
        rank = (row_indices == label).nonzero(as_tuple=False).item() + 1
        ranks.append(rank)
    return ranks


def load_triplets(file_path):
    """Load triplets from a tab-separated file."""
    triplets_df = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    return triplets_df


def get_one_hop_head_entity_neighbors(entity, triplets, max_degree=20):
    """Top max_degree one-hop neighbors for head entity based on degree."""
    entity_degrees = defaultdict(int)
    for head, relation, tail in triplets.values:
        entity_degrees[head] += 1
        entity_degrees[tail] += 1
    
    one_hop_neighbors = triplets[triplets['head'] == entity][['relation', 'tail']].values.tolist()
    sorted_neighbors = sorted(one_hop_neighbors, key=lambda x: entity_degrees[x[1]], reverse=True)
    top_neighbors = sorted_neighbors[:max_degree]
    
    return [f"{entity}-{rel}-{tail}" for rel, tail in top_neighbors]


def get_one_hop_tail_entity_neighbors(entity, triplets, max_degree=20):
    """Top max_degree one-hop neighbors for tail entity based on degree."""
    entity_degrees = defaultdict(int)
    for head, relation, tail in triplets.values:
        entity_degrees[head] += 1
        entity_degrees[tail] += 1
    
    one_hop_neighbors = triplets[triplets['tail'] == entity][['head', 'relation']].values.tolist()
    sorted_neighbors = sorted(one_hop_neighbors, key=lambda x: entity_degrees[x[0]], reverse=True)
    top_neighbors = sorted_neighbors[:max_degree]
    
    return [f"{head}-{rel}-{entity}" for head, rel in top_neighbors]


def save_evaluation_results(results, save_path):
    """Save final evaluation results as JSON."""
    import json
    os.makedirs(save_path, exist_ok=True)
    results_file = os.path.join(save_path, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_file}")


def hard_negative_hinge_loss(logits: torch.Tensor, labels: torch.Tensor,
                             top_k: int = 1, margin: float = 0.5, weight: float = 1.0) -> torch.Tensor:
    """Compute hinge loss on the hardest incorrect label(s) per sample.

    logits: (batch, num_classes), labels: (batch,)
    Returns scalar loss.
    """
    if top_k <= 0:
        return torch.tensor(0.0, device=logits.device)

    logits_clone = logits.clone()
    logits_clone[torch.arange(logits.size(0)), labels] = -1e9

    pos_scores = logits[torch.arange(logits.size(0)), labels]
    topk_vals, _ = torch.topk(logits_clone, top_k, dim=1)
    diffs = margin - (pos_scores.unsqueeze(1) - topk_vals)
    losses = torch.clamp(diffs, min=0.0)
    return weight * losses.mean()