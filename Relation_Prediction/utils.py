"""Utilities for negative-sample-free relation prediction training/evaluation."""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def safe_torch_load(path: str, map_location=None):
    """torch.load wrapper compatible with old/new PyTorch defaults."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def gpu_memory_mb(device) -> Optional[float]:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    return None


def _normalise_meta(meta):
    """Convert default-collated meta into list-of-dicts."""
    if meta is None:
        return []
    if isinstance(meta, list):
        return meta
    if isinstance(meta, dict):
        keys = list(meta.keys())
        if not keys:
            return []
        first = meta[keys[0]]
        if torch.is_tensor(first):
            n = first.size(0)
        else:
            n = len(first)
        rows = []
        for i in range(n):
            item = {}
            for key in keys:
                value = meta[key]
                if torch.is_tensor(value):
                    item[key] = value[i].item() if value.dim() > 0 else value.item()
                else:
                    item[key] = value[i]
            rows.append(item)
        return rows
    return []


# ==================== LOGIT ADJUSTMENTS ====================

def apply_type_relation_mask(
    logits: torch.Tensor,
    metas,
    type_pair_to_relation_indices: Optional[Mapping[Tuple[str, str], Sequence[int]]] = None,
) -> torch.Tensor:
    """Mask logits for relation labels invalid for a head/tail type pair."""
    if not type_pair_to_relation_indices:
        return logits
    rows = _normalise_meta(metas)
    if not rows:
        return logits

    masked_logits = logits.clone()
    num_labels = logits.size(-1)
    for i, row in enumerate(rows):
        type_pair = (row.get("head_type", "UNK"), row.get("tail_type", "UNK"))
        allowed = type_pair_to_relation_indices.get(type_pair)
        if not allowed or len(allowed) >= num_labels:
            continue
        mask = torch.full((num_labels,), -1e9, device=logits.device, dtype=logits.dtype)
        mask[list(allowed)] = 0.0
        masked_logits[i] = masked_logits[i] + mask
    return masked_logits


def apply_schema_relation_prior(
    logits: torch.Tensor,
    metas,
    schema_pair_relation_counts: Optional[Mapping[Tuple[str, str], Mapping[int, int]]] = None,
    weight: float = 0.1,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Add a soft train-only schema prior to relation logits.

    Unlike hard type masking, this never blocks a relation. It only nudges logits
    toward relations often observed for the induced head/tail schema pair.
    """
    if not schema_pair_relation_counts or weight <= 0:
        return logits
    rows = _normalise_meta(metas)
    if not rows:
        return logits

    adjusted = logits.clone()
    num_labels = logits.size(-1)
    for i, row in enumerate(rows):
        schema_pair = (row.get("head_schema", "UNK_SCHEMA"), row.get("tail_schema", "UNK_SCHEMA"))
        counts = schema_pair_relation_counts.get(schema_pair)
        if not counts:
            continue
        prior = torch.full((num_labels,), float(smoothing), device=logits.device, dtype=logits.dtype)
        for label_idx, count in counts.items():
            if 0 <= int(label_idx) < num_labels:
                prior[int(label_idx)] += float(count)
        prior = prior / prior.sum().clamp_min(1e-12)
        adjusted[i] = adjusted[i] + weight * torch.log(prior.clamp_min(1e-12))
    return adjusted


def apply_filtered_relation_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metas,
    known_true_relations_by_pair: Optional[Mapping[Tuple[str, str], Sequence[str]]],
    relation_to_idx: Mapping[str, int],
) -> torch.Tensor:
    """Mask other known true relations for the same (head, tail) pair.

    This is only for filtered evaluation. It follows the standard KGC idea that
    if multiple labels are true for a query, ranking another true label above the
    target should not be punished.
    """
    if not known_true_relations_by_pair:
        return logits
    rows = _normalise_meta(metas)
    if not rows:
        return logits
    filtered = logits.clone()
    for i, row in enumerate(rows):
        pair = (row.get("head"), row.get("tail"))
        true_rels = known_true_relations_by_pair.get(pair, [])
        gold = int(labels[i].item())
        for rel in true_rels:
            idx = relation_to_idx.get(rel)
            if idx is not None and idx != gold:
                filtered[i, idx] = -1e9
    return filtered


# ==================== RANKING / CLASSIFICATION METRICS ====================

def _rank_batch(logits: torch.Tensor, labels: torch.Tensor) -> Sequence[int]:
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    ranks = []
    for row_indices, label in zip(sorted_indices, labels):
        rank = (row_indices == label).nonzero(as_tuple=False).item() + 1
        ranks.append(rank)
    return ranks


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_labels: int,
    relation_counts: Optional[Mapping[int, int]] = None,
    ranks: Optional[Sequence[int]] = None,
    rare_threshold: int = 10,
    medium_threshold: int = 100,
) -> Dict[str, object]:
    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0

    per_label = {}
    f1s = []
    weighted_f1_sum = 0.0
    support_sum = 0
    for label in range(num_labels):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        f1 = _f1_from_counts(tp, fp, fn)
        per_label[label] = {"support": support, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        if support > 0:
            f1s.append(f1)
            weighted_f1_sum += f1 * support
            support_sum += support

    metrics = {
        "Accuracy": accuracy,
        "MacroF1": float(np.mean(f1s)) if f1s else 0.0,
        "WeightedF1": weighted_f1_sum / support_sum if support_sum else 0.0,
        "PerLabel": per_label,
    }

    if ranks is not None and relation_counts is not None:
        buckets = {"rare": [], "medium": [], "frequent": []}
        for r, label in zip(ranks, y_true):
            count = relation_counts.get(label, 0)
            if count <= rare_threshold:
                buckets["rare"].append(r)
            elif count <= medium_threshold:
                buckets["medium"].append(r)
            else:
                buckets["frequent"].append(r)
        bucket_metrics = {}
        for name, bucket_ranks in buckets.items():
            arr = np.array(bucket_ranks, dtype=np.float64)
            if len(arr) == 0:
                bucket_metrics[name] = {"count": 0, "MRR": None, "Hits@1": None, "Hits@3": None, "Hits@10": None}
            else:
                bucket_metrics[name] = {
                    "count": int(len(arr)),
                    "MRR": float(np.mean(1.0 / arr)),
                    "Hits@1": float(np.mean(arr <= 1)),
                    "Hits@3": float(np.mean(arr <= 3)),
                    "Hits@10": float(np.mean(arr <= 10)),
                }
        metrics["RelationFrequencyBuckets"] = bucket_metrics
    return metrics


def _aggregate_rank_metrics(ranks: Sequence[int], prefix: str = "") -> Dict[str, float]:
    arr = np.array(ranks, dtype=np.float64)
    p = prefix
    return {
        f"{p}MR": float(np.mean(arr)) if len(arr) else 0.0,
        f"{p}MRR": float(np.mean(1.0 / arr)) if len(arr) else 0.0,
        f"{p}Hits@1": float(np.mean(arr <= 1)) if len(arr) else 0.0,
        f"{p}Hits@3": float(np.mean(arr <= 3)) if len(arr) else 0.0,
        f"{p}Hits@5": float(np.mean(arr <= 5)) if len(arr) else 0.0,
        f"{p}Hits@10": float(np.mean(arr <= 10)) if len(arr) else 0.0,
    }


def evaluate_relation_model(
    model,
    dataloader,
    device,
    relation_to_idx: Mapping[str, int],
    test_triplets=None,
    tokenizer=None,
    graph_info=None,
    max_length: int = 128,
    type_pair_to_relation_indices: Optional[Mapping[Tuple[str, str], Sequence[int]]] = None,
    schema_pair_relation_counts: Optional[Mapping[Tuple[str, str], Mapping[int, int]]] = None,
    schema_prior_weight: float = 0.0,
    schema_prior_smoothing: float = 1.0,
    known_true_relations_by_pair: Optional[Mapping[Tuple[str, str], Sequence[str]]] = None,
    filtered_relation_eval: bool = False,
    relation_train_counts: Optional[Mapping[str, int]] = None,
    rare_threshold: int = 10,
    medium_threshold: int = 100,
):
    """Evaluate relation classifier with raw + optional filtered ranking metrics."""
    eval_model = unwrap_model(model)
    eval_model.eval()

    raw_rankings = []
    filtered_rankings = []
    y_true = []
    y_pred = []
    token_counts = []
    head_ctx_counts = []
    tail_ctx_counts = []
    pair_ctx_counts = []
    path_counts = []

    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                inputs, labels = batch
                metas = None
            else:
                inputs, labels, metas = batch
            inputs = {key: val.to(device, non_blocking=True) for key, val in inputs.items()}
            labels = labels.to(device, non_blocking=True)

            outputs = eval_model(**inputs)
            logits = outputs.logits
            logits = apply_schema_relation_prior(
                logits,
                metas,
                schema_pair_relation_counts,
                weight=schema_prior_weight,
                smoothing=schema_prior_smoothing,
            )
            logits = apply_type_relation_mask(logits, metas, type_pair_to_relation_indices)

            raw_rankings.extend(_rank_batch(logits, labels))
            ranking_logits = logits
            if filtered_relation_eval:
                ranking_logits = apply_filtered_relation_mask(
                    logits,
                    labels,
                    metas,
                    known_true_relations_by_pair,
                    relation_to_idx,
                )
                filtered_rankings.extend(_rank_batch(ranking_logits, labels))

            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

            for row in _normalise_meta(metas):
                if "text_token_count" in row:
                    token_counts.append(int(row["text_token_count"]))
                if "num_head_context_items" in row:
                    head_ctx_counts.append(int(row["num_head_context_items"]))
                if "num_tail_context_items" in row:
                    tail_ctx_counts.append(int(row["num_tail_context_items"]))
                if "num_pair_context_items" in row:
                    pair_ctx_counts.append(int(row["num_pair_context_items"]))
                if "num_path_items" in row:
                    path_counts.append(int(row["num_path_items"]))

    elapsed = time.perf_counter() - start

    relation_counts_by_idx = None
    if relation_train_counts:
        relation_counts_by_idx = {relation_to_idx[rel]: count for rel, count in relation_train_counts.items() if rel in relation_to_idx}

    cls_metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_labels=len(relation_to_idx),
        relation_counts=relation_counts_by_idx,
        ranks=filtered_rankings if filtered_relation_eval and filtered_rankings else raw_rankings,
        rare_threshold=rare_threshold,
        medium_threshold=medium_threshold,
    )

    results = _aggregate_rank_metrics(raw_rankings, prefix="")
    if filtered_relation_eval:
        results.update(_aggregate_rank_metrics(filtered_rankings, prefix="Filtered"))

    results.update({
        "Accuracy": cls_metrics["Accuracy"],
        "MacroF1": cls_metrics["MacroF1"],
        "WeightedF1": cls_metrics["WeightedF1"],
        "EvalSeconds": elapsed,
        "ExamplesPerSecond": len(y_true) / elapsed if elapsed > 0 else None,
        "AvgInputTokens": float(np.mean(token_counts)) if token_counts else None,
        "AvgHeadContextItems": float(np.mean(head_ctx_counts)) if head_ctx_counts else None,
        "AvgTailContextItems": float(np.mean(tail_ctx_counts)) if tail_ctx_counts else None,
        "AvgPairContextItems": float(np.mean(pair_ctx_counts)) if pair_ctx_counts else None,
        "AvgPathItems": float(np.mean(path_counts)) if path_counts else None,
        "RelationFrequencyBuckets": cls_metrics.get("RelationFrequencyBuckets", {}),
    })
    return results


# ==================== TRAINING LOSS ====================

def compute_relation_class_weights(
    train_triplets,
    relation_to_idx: Mapping[str, int],
    smoothing: float = 1.2,
    device=None,
) -> torch.Tensor:
    counts = Counter(train_triplets["relation"].tolist())
    weights = torch.ones(len(relation_to_idx), dtype=torch.float)
    for relation, idx in relation_to_idx.items():
        c = max(1, counts.get(relation, 0))
        weights[idx] = 1.0 / np.log(smoothing + c)
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights.to(device) if device is not None else weights


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, weight=weight, reduction="none", label_smoothing=label_smoothing)
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str = "ce",
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    if loss_type in {"weighted_ce", "deferred_weighted_ce"}:
        return F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=label_smoothing)
    if loss_type == "focal":
        return focal_loss(logits, labels, gamma=focal_gamma, weight=class_weights, label_smoothing=label_smoothing)
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def checkpoint_selection_score(metrics: Mapping[str, object], metric: str = "mrr", mrr_weight: float = 0.7, macro_f1_weight: float = 0.3) -> float:
    if metric == "macro_f1":
        return float(metrics.get("MacroF1", 0.0))
    if metric == "mrr_macro_f1":
        mrr = float(metrics.get("FilteredMRR", metrics.get("MRR", 0.0)) or 0.0)
        macro_f1 = float(metrics.get("MacroF1", 0.0) or 0.0)
        return mrr_weight * mrr + macro_f1_weight * macro_f1
    return float(metrics.get("FilteredMRR", metrics.get("MRR", 0.0)) or 0.0)


# ==================== OUTPUT ====================

def save_test_results(epoch: int, test_results: Mapping[str, object], save_path: str, task: str = "relation") -> None:
    """Append a tabular metrics row and a JSONL metrics row."""
    os.makedirs(save_path, exist_ok=True)

    txt_path = os.path.join(save_path, f"{task}_test_results.txt")
    header = (
        "Epoch\tMR\tMRR\tHits@1\tHits@3\tHits@5\tHits@10\t"
        "FilteredMRR\tFilteredHits@1\tFilteredHits@3\tFilteredHits@10\t"
        "Accuracy\tMacroF1\tWeightedF1\tEvalSeconds\tExamplesPerSecond\tAvgInputTokens\n"
    )
    if not os.path.exists(txt_path):
        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(header)
    row = (
        f"{epoch + 1}\t{test_results.get('MR', 0):.2f}\t{test_results.get('MRR', 0):.4f}\t"
        f"{test_results.get('Hits@1', 0):.4f}\t{test_results.get('Hits@3', 0):.4f}\t"
        f"{test_results.get('Hits@5', 0):.4f}\t{test_results.get('Hits@10', 0):.4f}\t"
        f"{test_results.get('FilteredMRR') or 0:.4f}\t{test_results.get('FilteredHits@1') or 0:.4f}\t"
        f"{test_results.get('FilteredHits@3') or 0:.4f}\t{test_results.get('FilteredHits@10') or 0:.4f}\t"
        f"{test_results.get('Accuracy', 0):.4f}\t{test_results.get('MacroF1', 0):.4f}\t"
        f"{test_results.get('WeightedF1', 0):.4f}\t{test_results.get('EvalSeconds', 0):.2f}\t"
        f"{test_results.get('ExamplesPerSecond') or 0:.2f}\t{test_results.get('AvgInputTokens') or 0:.2f}\n"
    )
    with open(txt_path, "a", encoding="utf-8") as file:
        file.write(row)

    json_path = os.path.join(save_path, f"{task}_metrics.jsonl")
    json_row = {"epoch": epoch + 1, **dict(test_results)}
    with open(json_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(json_row, ensure_ascii=False, default=str) + "\n")


def save_checkpoint(model, optimizer, epoch: int, checkpoint_path: str) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model_state = unwrap_model(model).state_dict()
    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path: str, map_location=None) -> int:
    if not os.path.exists(checkpoint_path):
        return 0
    checkpoint = safe_torch_load(checkpoint_path, map_location=map_location)
    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))
