"""Train negative-sample-free relation-prediction models for general KG benchmarks.

Main features:
- no negative sampling or corrupted triples,
- train-only graph context and target-triple exclusion,
- generic KG text normalization and optional label maps,
- adaptive/grouped/pair-specific context,
- optional compressed path motifs,
- induced schema tokens and soft schema priors,
- dynamic padding for speed/memory,
- positive-only relation-balanced sampler and deferred weighted CE,
- raw and filtered relation-ranking evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from collections import Counter
from typing import Dict, Mapping

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

import config
from data_loader import (
    KGRelationDataset,
    build_known_true_relations_by_pair,
    build_relation_mask_by_type_pair,
    build_relation_to_idx,
    build_schema_prior_counts,
    load_entity_types,
    load_label_map,
    load_triplets,
    make_relation_collate_fn,
    precompute_graph_info,
    validate_relation_coverage,
)
from model import get_model_and_tokenizer
from utils import (
    apply_schema_relation_prior,
    apply_type_relation_mask,
    checkpoint_selection_score,
    compute_loss,
    compute_relation_class_weights,
    evaluate_relation_model,
    gpu_memory_mb,
    load_checkpoint,
    save_checkpoint,
    save_json,
    save_test_results,
    safe_torch_load,
    set_seed,
)

warnings.filterwarnings("ignore")


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = config.MASTER_PORT
    dist.init_process_group(config.DDP_BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = not getattr(config, "DETERMINISTIC", False)


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def dataloader_kwargs(train: bool) -> Dict[str, object]:
    workers = int(getattr(config, "NUM_WORKERS", 0))
    kwargs = {
        "num_workers": workers,
        "pin_memory": bool(getattr(config, "PIN_MEMORY", True)) and torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(train)
        kwargs["prefetch_factor"] = int(getattr(config, "PREFETCH_FACTOR", 2))
    return kwargs


def dataset_kwargs() -> Dict[str, object]:
    return {
        "pretokenize": bool(getattr(config, "PRETOKENIZE", False)),
        "dynamic_padding": bool(getattr(config, "DYNAMIC_PADDING", True)),
        "default_entity_type": getattr(config, "DEFAULT_ENTITY_TYPE", "UNK"),
        "normalize_entity_text": bool(getattr(config, "NORMALIZE_ENTITY_TEXT", True)),
        "normalize_relation_text": bool(getattr(config, "NORMALIZE_RELATION_TEXT", True)),
        "context_mode": getattr(config, "CONTEXT_MODE", "adaptive"),
        "min_context": int(getattr(config, "MIN_CONTEXT", 5)),
        "base_context": int(getattr(config, "BASE_CONTEXT", 15)),
        "max_context": int(getattr(config, "MAX_CONTEXT", getattr(config, "MAX_DEGREE", 30))),
        "max_head_context_tokens": int(getattr(config, "MAX_HEAD_CONTEXT_TOKENS", 40)),
        "max_tail_context_tokens": int(getattr(config, "MAX_TAIL_CONTEXT_TOKENS", 40)),
        "max_pair_context_tokens": int(getattr(config, "MAX_PAIR_CONTEXT_TOKENS", 24)),
        "max_path_context_tokens": int(getattr(config, "MAX_PATH_CONTEXT_TOKENS", 20)),
        "use_second_hop_context": bool(getattr(config, "USE_SECOND_HOP_CONTEXT", False)),
        "second_hop_budget": int(getattr(config, "SECOND_HOP_BUDGET", 5)),
        "use_entity_types": bool(getattr(config, "USE_ENTITY_TYPES", False)),
        "use_schema_tokens": bool(getattr(config, "USE_SCHEMA_TOKENS", True)),
        "use_pair_context": bool(getattr(config, "USE_PAIR_CONTEXT", True)),
        "max_common_neighbors": int(getattr(config, "MAX_COMMON_NEIGHBORS", 5)),
        "max_pair_motifs": int(getattr(config, "MAX_PAIR_MOTIFS", 5)),
        "use_path_context": bool(getattr(config, "USE_PATH_CONTEXT", False)),
        "path_context_mode": getattr(config, "PATH_CONTEXT_MODE", "motif"),
        "max_paths": int(getattr(config, "MAX_PATHS", 3)),
        "group_context_by_relation": bool(getattr(config, "GROUP_CONTEXT_BY_RELATION", True)),
        "max_neighbors_per_relation": int(getattr(config, "MAX_NEIGHBORS_PER_RELATION", 3)),
        "w_degree": float(getattr(config, "W_DEGREE", 1.0)),
        "w_rel_rarity": float(getattr(config, "W_REL_RARITY", 0.35)),
        "w_redundancy": float(getattr(config, "W_REDUNDANCY", 0.15)),
        "use_context_dropout": bool(getattr(config, "USE_CONTEXT_DROPOUT", False)),
        "context_dropout_prob": float(getattr(config, "CONTEXT_DROPOUT_PROB", 0.0)),
        "shuffle_context_items": bool(getattr(config, "SHUFFLE_CONTEXT_ITEMS", False)),
    }


def _file_fingerprint(path: object) -> Dict[str, object]:
    if not path:
        return {"path": None, "exists": False}
    path_str = str(path)
    if not os.path.exists(path_str):
        return {"path": path_str, "exists": False}
    stat = os.stat(path_str)
    return {
        "path": os.path.abspath(path_str),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def graph_cache_metadata(dataset_config: Mapping[str, object]) -> Dict[str, object]:
    entity_type_file = dataset_config.get("ENTITY_TYPE_FILE") or getattr(config, "ENTITY_TYPE_FILE", None)
    return {
        "version": getattr(config, "GRAPH_CACHE_VERSION", "v1"),
        "train_file": _file_fingerprint(dataset_config.get("TRAIN_FILE_PATH")),
        "entity_type_file": _file_fingerprint(entity_type_file),
        "default_entity_type": getattr(config, "DEFAULT_ENTITY_TYPE", "UNK"),
        "use_entity_types": bool(getattr(config, "USE_ENTITY_TYPES", False)),
        "schema_top_relations": int(getattr(config, "SCHEMA_TOP_RELATIONS", 3)),
        "schema_mode": getattr(config, "SCHEMA_MODE", "relation_signature"),
    }


def graph_cache_path(dataset_config: Mapping[str, object]) -> str:
    metadata = graph_cache_metadata(dataset_config)
    digest = hashlib.sha1(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    name = f"{config.GRAPH_CACHE_PREFIX}_{digest}.pt"
    return os.path.join(dataset_config["MODEL_SAVE_PATH"], name)


def load_or_build_graph_info(rank: int, distributed: bool, train_triplets, entity_types, dataset_config):
    cache_path = graph_cache_path(dataset_config)
    expected_metadata = graph_cache_metadata(dataset_config)
    graph_info = None

    if rank == 0:
        os.makedirs(dataset_config["MODEL_SAVE_PATH"], exist_ok=True)
        if os.path.exists(cache_path):
            payload = safe_torch_load(cache_path, map_location="cpu")
            if isinstance(payload, dict) and payload.get("metadata") == expected_metadata and "graph_info" in payload:
                print(f"Loading validated train-only graph cache: {cache_path}")
                graph_info = payload["graph_info"]
            else:
                print(f"Ignoring stale or legacy graph cache: {cache_path}")

        if graph_info is None:
            print("Precomputing train-only graph information...")
            start = time.perf_counter()
            graph_info = precompute_graph_info(
                train_triplets,
                entity_types=entity_types,
                default_type=getattr(config, "DEFAULT_ENTITY_TYPE", "UNK"),
                schema_top_relations=int(getattr(config, "SCHEMA_TOP_RELATIONS", 3)),
                schema_mode=getattr(config, "SCHEMA_MODE", "relation_signature"),
            )
            torch.save({"metadata": expected_metadata, "graph_info": graph_info}, cache_path)
            print(f"Graph precompute complete in {time.perf_counter() - start:.2f}s. Saved to {cache_path}")

    if distributed:
        dist.barrier()

    if rank != 0:
        payload = safe_torch_load(cache_path, map_location="cpu")
        graph_info = payload["graph_info"] if isinstance(payload, dict) and "graph_info" in payload else payload

    if distributed:
        dist.barrier()
    return graph_info


def _positive_relation_sampler(train_triplets, relation_to_idx):
    counts = Counter(train_triplets["relation"].tolist())
    power = float(getattr(config, "RELATION_SAMPLER_POWER", 0.5))
    weights = [1.0 / (max(1, counts[str(row.relation)]) ** power) for row in train_triplets.itertuples(index=False)]
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def create_datasets_and_loaders(
    rank: int,
    world_size: int,
    distributed: bool,
    tokenizer,
    relation_to_idx,
    graph_info,
    entity_types,
    entity_label_map,
    relation_label_map,
    dataset_name,
    train_triplets,
    valid_triplets,
    test_triplets,
):
    kwargs = dataset_kwargs()
    train_dataset = KGRelationDataset(
        train_triplets,
        tokenizer,
        relation_to_idx,
        graph_info,
        entity_types,
        entity_label_map=entity_label_map,
        relation_label_map=relation_label_map,
        dataset_name=dataset_name,
        max_length=config.MAX_LENGTH,
        training=True,
        **kwargs,
    )
    valid_dataset = KGRelationDataset(
        valid_triplets,
        tokenizer,
        relation_to_idx,
        graph_info,
        entity_types,
        entity_label_map=entity_label_map,
        relation_label_map=relation_label_map,
        dataset_name=dataset_name,
        max_length=config.MAX_LENGTH,
        training=False,
        **kwargs,
    )
    test_dataset = KGRelationDataset(
        test_triplets,
        tokenizer,
        relation_to_idx,
        graph_info,
        entity_types,
        entity_label_map=entity_label_map,
        relation_label_map=relation_label_map,
        dataset_name=dataset_name,
        max_length=config.MAX_LENGTH,
        training=False,
        **kwargs,
    )

    train_sampler = None
    shuffle = True
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
        if getattr(config, "TRAIN_SAMPLER", "random") == "relation_balanced" and rank == 0:
            print("Relation-balanced sampler is disabled under DDP; using DistributedSampler.")
    elif getattr(config, "TRAIN_SAMPLER", "random") == "relation_balanced":
        train_sampler = _positive_relation_sampler(train_triplets, relation_to_idx)
        shuffle = False

    collate_fn = make_relation_collate_fn(
        tokenizer,
        dynamic_padding=bool(getattr(config, "DYNAMIC_PADDING", True)),
        max_length=config.MAX_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.PER_GPU_BATCH_SIZE if distributed else config.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **dataloader_kwargs(train=True),
    )
    eval_batch = config.PER_GPU_BATCH_SIZE if distributed else config.BATCH_SIZE
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch, shuffle=False, collate_fn=collate_fn, **dataloader_kwargs(train=False))
    test_loader = DataLoader(test_dataset, batch_size=eval_batch, shuffle=False, collate_fn=collate_fn, **dataloader_kwargs(train=False))
    return train_loader, valid_loader, test_loader, train_sampler


def print_metrics(prefix: str, metrics: Mapping[str, object]) -> None:
    print(prefix)
    print(f"  MR: {metrics['MR']:.2f}")
    print(f"  MRR: {metrics['MRR']:.4f}")
    print(f"  Hits@1: {metrics['Hits@1']:.4f}")
    print(f"  Hits@3: {metrics['Hits@3']:.4f}")
    print(f"  Hits@5: {metrics['Hits@5']:.4f}")
    print(f"  Hits@10: {metrics['Hits@10']:.4f}")
    if "FilteredMRR" in metrics:
        print(f"  Filtered MRR: {metrics['FilteredMRR']:.4f}")
        print(f"  Filtered Hits@1/3/10: {metrics['FilteredHits@1']:.4f} / {metrics['FilteredHits@3']:.4f} / {metrics['FilteredHits@10']:.4f}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Macro-F1: {metrics['MacroF1']:.4f}")
    print(f"  Weighted-F1: {metrics['WeightedF1']:.4f}")
    if metrics.get("ExamplesPerSecond"):
        print(f"  Examples/s: {metrics['ExamplesPerSecond']:.2f}")
    if metrics.get("AvgInputTokens") is not None:
        print(f"  Avg input tokens: {metrics['AvgInputTokens']:.2f}")
    if metrics.get("AvgPairContextItems") is not None:
        print(f"  Avg pair context items: {metrics['AvgPairContextItems']:.2f}")
    print()


def _effective_loss_type(epoch: int) -> str:
    loss_type = getattr(config, "LOSS_TYPE", "ce")
    if loss_type == "deferred_weighted_ce" and epoch < int(getattr(config, "REWEIGHT_START_EPOCH", 3)):
        return "ce"
    return loss_type


def train_impl(rank: int, world_size: int, dataset_config: Mapping[str, object], distributed: bool) -> None:
    if distributed:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = config.DEVICE

    set_seed(config.SEED + rank, deterministic=getattr(config, "DETERMINISTIC", False))
    os.makedirs(dataset_config["MODEL_SAVE_PATH"], exist_ok=True)

    if rank == 0:
        print("=" * 80)
        print("Knowledge Graph Completion - General Relation Prediction")
        print(f"DATASET: {dataset_config['name']}")
        print("=" * 80)

    train_triplets = load_triplets(dataset_config["TRAIN_FILE_PATH"])
    valid_triplets = load_triplets(dataset_config["VALID_FILE_PATH"])
    test_triplets = load_triplets(dataset_config["TEST_FILE_PATH"])

    relation_to_idx = build_relation_to_idx(train_triplets)
    validate_relation_coverage(relation_to_idx, valid_triplets, test_triplets)
    num_relations = len(relation_to_idx)
    relation_train_counts = dict(Counter(train_triplets["relation"].tolist()))
    known_true_relations_by_pair = build_known_true_relations_by_pair(train_triplets, valid_triplets, test_triplets)

    entity_type_file = dataset_config.get("ENTITY_TYPE_FILE") or getattr(config, "ENTITY_TYPE_FILE", None)
    entity_types = load_entity_types(entity_type_file, default_type=getattr(config, "DEFAULT_ENTITY_TYPE", "UNK"))
    entity_label_map = load_label_map(dataset_config.get("ENTITY_LABEL_FILE")) if getattr(config, "USE_LABEL_MAPS", True) else {}
    relation_label_map = load_label_map(dataset_config.get("RELATION_LABEL_FILE")) if getattr(config, "USE_LABEL_MAPS", True) else {}

    graph_info = load_or_build_graph_info(rank, distributed, train_triplets, entity_types, dataset_config)

    type_pair_to_relation_indices = None
    if getattr(config, "USE_TYPE_RELATION_MASK", False):
        type_pair_to_relation_indices = build_relation_mask_by_type_pair(graph_info.get("type_pair_to_relations", {}), relation_to_idx)

    schema_pair_relation_counts = None
    if getattr(config, "USE_SCHEMA_PRIOR", True):
        schema_pair_relation_counts = build_schema_prior_counts(graph_info, relation_to_idx)

    model, tokenizer = get_model_and_tokenizer(
        config.MODEL_NAME,
        num_labels=num_relations,
        special_tokens=getattr(config, "SPECIAL_TOKENS", None),
    )
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    if rank == 0:
        run_config = {
            "dataset": dict(dataset_config),
            "model_name": config.MODEL_NAME,
            "num_relations": num_relations,
            "relation_to_idx": relation_to_idx,
            "context": dataset_kwargs(),
            "loss_type": config.LOSS_TYPE,
            "label_smoothing": config.LABEL_SMOOTHING,
            "train_sampler": getattr(config, "TRAIN_SAMPLER", "random"),
            "dynamic_padding": getattr(config, "DYNAMIC_PADDING", True),
            "filtered_relation_eval": getattr(config, "FILTERED_RELATION_EVAL", True),
            "use_schema_prior": getattr(config, "USE_SCHEMA_PRIOR", True),
            "schema_prior_weight": getattr(config, "SCHEMA_PRIOR_WEIGHT", 0.1),
            "entity_label_map_size": len(entity_label_map),
            "relation_label_map_size": len(relation_label_map),
        }
        save_json(run_config, os.path.join(dataset_config["MODEL_SAVE_PATH"], config.CONFIG_JSON))

    train_loader, valid_loader, test_loader, train_sampler = create_datasets_and_loaders(
        rank,
        world_size,
        distributed,
        tokenizer,
        relation_to_idx,
        graph_info,
        entity_types,
        entity_label_map,
        relation_label_map,
        dataset_config.get("name", ""),
        train_triplets,
        valid_triplets,
        test_triplets,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    class_weights = None
    if config.LOSS_TYPE in {"weighted_ce", "deferred_weighted_ce", "focal"}:
        class_weights = compute_relation_class_weights(
            train_triplets,
            relation_to_idx,
            smoothing=getattr(config, "CLASS_WEIGHT_SMOOTHING", 1.2),
            device=device,
        )

    best_selection_score = float("-inf")
    best_val_mrr = float("-inf")
    epochs_without_improvement = 0
    best_checkpoint_path = os.path.join(dataset_config["MODEL_SAVE_PATH"], "checkpoint_best.pth")
    gradient_accumulation_steps = max(1, int(getattr(config, "GRADIENT_ACCUMULATION_STEPS", 1)))

    if rank == 0:
        print(f"Starting training for {config.NUM_EPOCHS} epochs on {dataset_config['name']}...")
        print(f"Relations: {num_relations}")
        print(f"Train/Valid/Test triples: {len(train_triplets)}/{len(valid_triplets)}/{len(test_triplets)}")
        print(f"Loss: {config.LOSS_TYPE}, label_smoothing={config.LABEL_SMOOTHING}")
        print(f"Entity labels loaded: {len(entity_label_map)}, relation labels loaded: {len(relation_label_map)}")

    for epoch in range(config.NUM_EPOCHS):
        if distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        total_batches = len(train_loader)
        start_epoch = time.perf_counter()
        effective_loss_type = _effective_loss_type(epoch)

        for batch_idx, (inputs, labels, metas) in enumerate(train_loader):
            inputs = {key: val.to(device, non_blocking=True) for key, val in inputs.items()}
            labels = labels.to(device, non_blocking=True)

            outputs = model(**inputs)
            logits = outputs.logits
            logits = apply_schema_relation_prior(
                logits,
                metas,
                schema_pair_relation_counts,
                weight=float(getattr(config, "SCHEMA_PRIOR_WEIGHT", 0.1)) if getattr(config, "USE_SCHEMA_PRIOR", True) else 0.0,
                smoothing=float(getattr(config, "SCHEMA_PRIOR_SMOOTHING", 1.0)),
            )

            if getattr(config, "APPLY_TYPE_MASK_DURING_TRAINING", False):
                logits = apply_type_relation_mask(logits, metas, type_pair_to_relation_indices)

            loss = compute_loss(
                logits,
                labels,
                loss_type=effective_loss_type,
                class_weights=class_weights,
                label_smoothing=float(getattr(config, "LABEL_SMOOTHING", 0.0)),
                focal_gamma=float(getattr(config, "FOCAL_GAMMA", 2.0)),
            )

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (batch_idx + 1) == total_batches and (batch_idx + 1) % gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * gradient_accumulation_steps
            if rank == 0 and (batch_idx + 1) % 100 == 0:
                avg = train_loss / (batch_idx + 1)
                print(
                    f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{total_batches} | "
                    f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | Avg Loss: {avg:.4f}"
                )

        epoch_seconds = time.perf_counter() - start_epoch
        avg_train_loss = train_loss / max(1, total_batches)

        stop_now = False
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(
                f"Epoch {epoch + 1}/{config.NUM_EPOCHS} - Avg Train Loss: {avg_train_loss:.4f} "
                f"- LossMode: {effective_loss_type} - Time: {epoch_seconds:.2f}s"
            )
            print(f"{'=' * 60}")
            print("Evaluating on validation set...")
            valid_results = evaluate_relation_model(
                model,
                valid_loader,
                device,
                relation_to_idx,
                graph_info=graph_info,
                max_length=config.MAX_LENGTH,
                type_pair_to_relation_indices=type_pair_to_relation_indices if config.USE_TYPE_RELATION_MASK else None,
                schema_pair_relation_counts=schema_pair_relation_counts if config.USE_SCHEMA_PRIOR else None,
                schema_prior_weight=float(getattr(config, "SCHEMA_PRIOR_WEIGHT", 0.1)),
                schema_prior_smoothing=float(getattr(config, "SCHEMA_PRIOR_SMOOTHING", 1.0)),
                known_true_relations_by_pair=known_true_relations_by_pair,
                filtered_relation_eval=bool(getattr(config, "FILTERED_RELATION_EVAL", True)),
                relation_train_counts=relation_train_counts,
                rare_threshold=int(getattr(config, "RARE_RELATION_THRESHOLD", 10)),
                medium_threshold=int(getattr(config, "MEDIUM_RELATION_THRESHOLD", 100)),
            )
            valid_results["TrainLoss"] = avg_train_loss
            valid_results["TrainEpochSeconds"] = epoch_seconds
            valid_results["MaxGpuMemoryMB"] = gpu_memory_mb(device)
            valid_results["EffectiveLossType"] = effective_loss_type
            print_metrics("Validation results:", valid_results)
            save_test_results(epoch, valid_results, dataset_config["MODEL_SAVE_PATH"], task="relation_val")

            if getattr(config, "SAVE_EVERY_EPOCH", False):
                save_checkpoint(model, optimizer, epoch + 1, os.path.join(dataset_config["MODEL_SAVE_PATH"], f"checkpoint_epoch_{epoch + 1}.pth"))

            current_score = checkpoint_selection_score(
                valid_results,
                metric=getattr(config, "CHECKPOINT_METRIC", "mrr"),
                mrr_weight=float(getattr(config, "CHECKPOINT_MRR_WEIGHT", 0.7)),
                macro_f1_weight=float(getattr(config, "CHECKPOINT_MACRO_F1_WEIGHT", 0.3)),
            )
            current_mrr = float(valid_results.get("FilteredMRR", valid_results.get("MRR", 0.0)) or 0.0)
            improved = current_score > (best_selection_score + getattr(config, "EARLY_STOPPING_MIN_DELTA", 0.0))
            if improved:
                best_selection_score = current_score
                best_val_mrr = current_mrr
                epochs_without_improvement = 0
                save_checkpoint(model, optimizer, epoch + 1, best_checkpoint_path)
                print(f"Best validation score improved to {best_selection_score:.4f}; MRR={best_val_mrr:.4f}; saved checkpoint_best.pth")
            else:
                epochs_without_improvement += 1
                print(
                    f"No validation score improvement for {epochs_without_improvement} epoch(s) "
                    f"(patience={config.EARLY_STOPPING_PATIENCE})."
                )

            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                stop_now = True

        if distributed:
            stop_tensor = torch.tensor([1 if stop_now else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_now = bool(stop_tensor.item())
        if stop_now:
            break

    if rank == 0:
        if os.path.exists(best_checkpoint_path):
            load_checkpoint(model, optimizer=None, checkpoint_path=best_checkpoint_path, map_location=device)
        print("\nTraining completed. Evaluating best checkpoint on test set...")
        test_results = evaluate_relation_model(
            model,
            test_loader,
            device,
            relation_to_idx,
            graph_info=graph_info,
            max_length=config.MAX_LENGTH,
            type_pair_to_relation_indices=type_pair_to_relation_indices if config.USE_TYPE_RELATION_MASK else None,
            schema_pair_relation_counts=schema_pair_relation_counts if config.USE_SCHEMA_PRIOR else None,
            schema_prior_weight=float(getattr(config, "SCHEMA_PRIOR_WEIGHT", 0.1)),
            schema_prior_smoothing=float(getattr(config, "SCHEMA_PRIOR_SMOOTHING", 1.0)),
            known_true_relations_by_pair=known_true_relations_by_pair,
            filtered_relation_eval=bool(getattr(config, "FILTERED_RELATION_EVAL", True)),
            relation_train_counts=relation_train_counts,
            rare_threshold=int(getattr(config, "RARE_RELATION_THRESHOLD", 10)),
            medium_threshold=int(getattr(config, "MEDIUM_RELATION_THRESHOLD", 100)),
        )
        test_results["MaxGpuMemoryMB"] = gpu_memory_mb(device)
        print_metrics("Test results:", test_results)
        save_test_results(config.NUM_EPOCHS - 1, test_results, dataset_config["MODEL_SAVE_PATH"], task="relation")

        save_model = model.module if hasattr(model, "module") else model
        save_model.save_pretrained(dataset_config["MODEL_SAVE_PATH"])
        tokenizer.save_pretrained(dataset_config["MODEL_SAVE_PATH"])

    if distributed:
        dist.barrier()
        cleanup_ddp()


def train_single_process(dataset_config: Mapping[str, object]) -> None:
    train_impl(rank=0, world_size=1, dataset_config=dataset_config, distributed=False)


def train_ddp(rank: int, world_size: int, dataset_config: Mapping[str, object]) -> None:
    train_impl(rank=rank, world_size=world_size, dataset_config=dataset_config, distributed=True)


if __name__ == "__main__":
    print("=" * 100)
    print("KNOWLEDGE GRAPH COMPLETION – GENERAL RELATION PREDICTION")
    print("=" * 100)
    config.print_config_summary()

    for idx, dataset in enumerate(config.DATASETS):
        print(f"\n{'=' * 100}")
        print(f"DATASET {idx + 1}/{len(config.DATASETS)} → {dataset['name'].upper()}")
        print(f"Model will be saved to: {dataset['MODEL_SAVE_PATH']}")
        print(f"{'=' * 100}\n")

        if config.NUM_GPUS > 1:
            mp.spawn(train_ddp, args=(config.NUM_GPUS, dataset), nprocs=config.NUM_GPUS, join=True)
        else:
            train_single_process(dataset)

    print("\nALL DATASETS PROCESSED SUCCESSFULLY!")
