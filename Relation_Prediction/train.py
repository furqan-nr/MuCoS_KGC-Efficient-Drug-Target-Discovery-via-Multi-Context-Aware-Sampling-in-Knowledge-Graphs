import os
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import warnings

import config
from data_loader import load_triplets, precompute_entity_info, KGRelationDataset
from model import get_model_and_tokenizer
from utils import evaluate_relation_model, save_test_results, save_checkpoint, hard_negative_hinge_loss
import torch.nn.functional as F
from negative_sampler import NegativeSampler

warnings.filterwarnings("ignore")



# ==================== DDP SETUP AND CLEANUP ====================
# These functions set up the distributed training environment for PyTorch.
# - `setup(rank, world_size)`: Initializes process groups, sets GPU device for this rank,
#   and enables cuDNN benchmark for better performance.
# - `cleanup()`: Destroys the process group after training to release resources.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.MASTER_PORT
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True


def cleanup():
    dist.destroy_process_group()



# ==================== TRAIN FUNCTION ====================
# This is the main training loop executed by each process in DDP.
# It handles:
# - Loading dataset triplets
# - Precomputing graph context for entities (neighbors)
# - Initializing the model and wrapping it with DistributedDataParallel (DDP)
# - Creating DataLoaders with DistributedSampler for proper sharding across GPUs
# - Gradient accumulation for memory efficiency
# - Epoch-wise training, logging, evaluation, and checkpointing

def train(rank, world_size, dataset_config):
    """DDP training function – receives the current dataset config"""
    setup(rank, world_size)
    os.makedirs(dataset_config["MODEL_SAVE_PATH"], exist_ok=True)
    entity_neighbors_path = os.path.join(dataset_config["MODEL_SAVE_PATH"], "entity_neighbors_train_only.pt")

    if rank == 0:
        print("=" * 70)
        print(f"Knowledge Graph Completion - Relation Prediction")
        print(f"DATASET: {dataset_config['name']}")
        print("=" * 70)

    # Load triplets using the passed dataset config
    train_triplets = load_triplets(dataset_config["TRAIN_FILE_PATH"])
    valid_triplets = load_triplets(dataset_config["VALID_FILE_PATH"])
    test_triplets = load_triplets(dataset_config["TEST_FILE_PATH"])

    # Build relation mapping
    all_relations = pd.concat([train_triplets['relation'],
                               valid_triplets['relation'],
                               test_triplets['relation']]).unique().tolist()
    relation_to_idx = {rel: idx for idx, rel in enumerate(all_relations)}
    num_relations = len(relation_to_idx)

    # Graph context is built from training triples only to avoid leakage.


        # -------------------- PRECOMPUTE ENTITY INFO --------------------
    # Precomputes neighbor information for all entities and shares it
    # with other processes via temporary file to avoid redundant computation.

    if rank == 0:
        if os.path.exists(entity_neighbors_path):
            print(f"Loading cached entity information from {entity_neighbors_path}...")
            entity_incoming_neighbors = torch.load(entity_neighbors_path)
        else:
            print("Precomputing entity information...")
            entity_degrees, entity_incoming_neighbors = precompute_entity_info(train_triplets, config.MAX_DEGREE)
            torch.save(entity_incoming_neighbors, entity_neighbors_path)
            print(f"Precomputation complete. Saved to {entity_neighbors_path}")

    dist.barrier()

    if rank != 0:
        entity_incoming_neighbors = torch.load(entity_neighbors_path)

    dist.barrier()

    if rank == 0:
        print("All ranks have loaded precomputed data.")

    # -------------------- NEGATIVE SAMPLER (POOL CACHE) --------------------
    # Build or load a per-entity negative candidate pool to support
    # future hard-negative mining strategies (diffusion, cached lookups, etc.).
    neg_sampler_cache = os.path.join(dataset_config["MODEL_SAVE_PATH"], "neg_sampler_pool.pt")
    neg_sampler = NegativeSampler(dataset_config["MODEL_SAVE_PATH"], pool_size=getattr(config, 'HNM_POOL_SIZE', 100))

    if rank == 0:
        if os.path.exists(neg_sampler_cache):
            print(f"Loading negative-sampler cache from {neg_sampler_cache}...")
            neg_sampler.load(neg_sampler_cache)
        else:
            print("Building negative sampler pools (this may take a moment)...")
            neg_sampler.build_from_triplets(train_triplets, entity_incoming_neighbors, pool_size=getattr(config, 'HNM_POOL_SIZE', 100))
            neg_sampler.save(neg_sampler_cache)

    dist.barrier()

    # Ensure all ranks load the same sampler mapping
    if rank != 0:
        neg_sampler.load(neg_sampler_cache)

    dist.barrier()

    if rank == 0:
        print("Negative sampler ready.")



        # -------------------- MODEL INITIALIZATION --------------------
    # Load the selected transformer model and tokenizer, move model to GPU,
    # and wrap it in DistributedDataParallel for multi-GPU training.

    model, tokenizer = get_model_and_tokenizer(config.MODEL_NAME, num_labels=num_relations)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)



        # -------------------- DATALOADER SETUP --------------------
    # Training dataset uses DistributedSampler for proper data sharding.
    # Validation and test datasets use standard DataLoader.

    train_dataset = KGRelationDataset(train_triplets, tokenizer, relation_to_idx,
                                      entity_incoming_neighbors, max_length=config.MAX_LENGTH)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.PER_GPU_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # Validation / Test dataloaders
    valid_dataset = KGRelationDataset(valid_triplets, tokenizer, relation_to_idx,
                                      entity_incoming_neighbors, max_length=config.MAX_LENGTH)
    test_dataset = KGRelationDataset(test_triplets, tokenizer, relation_to_idx,
                                     entity_incoming_neighbors, max_length=config.MAX_LENGTH)

    valid_dataloader = DataLoader(valid_dataset, batch_size=config.PER_GPU_BATCH_SIZE,
                                  shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.PER_GPU_BATCH_SIZE,
                                 shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    gradient_accumulation_steps = 2
    best_val_mrr = float("-inf")
    epochs_without_improvement = 0
    best_checkpoint_path = os.path.join(dataset_config["MODEL_SAVE_PATH"], "checkpoint_best.pth")

    if rank == 0:
        print(f"Starting training for {config.NUM_EPOCHS} epochs on {dataset_config['name']}...")



        # -------------------- TRAINING LOOP --------------------
    # For each epoch, iterate over batches, compute loss, and update gradients.
    # Gradient accumulation is used to simulate larger batch sizes.

    for epoch in range(config.NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        total_batches = len(train_dataloader)

        for batch_idx, (inputs, labels, meta) in enumerate(train_dataloader):
            inputs = {k: v.to(rank, non_blocking=True) for k, v in inputs.items()}
            labels = labels.to(rank, non_blocking=True)

            # forward to get logits, compute CE loss manually so we can add hinge term
            outputs = model(**inputs)
            logits = outputs.logits
            loss_ce = F.cross_entropy(logits, labels)

            if config.HARD_NEGATIVE_MINING_ENABLED:
                # Use sampler pools to provide candidate negative relations indirectly
                # by constructing negative-inputs for sampled negative tails (entity-level).
                # Build a small batch of negative examples per sample by replacing the tail
                # with sampled negatives from neg_sampler and scoring their logits.
                batch_negatives = []
                for i, m in enumerate(meta):
                    tail = m.get('tail')
                    # sample k negative tails for this entity
                    sampled_tails = neg_sampler.sample(tail, k=getattr(config, 'HNM_ENTITY_NEG_K', 1))
                    batch_negatives.append(sampled_tails)

                # If no entity negatives found, fallback to logits-based hinge
                has_entity_neg = any(len(x) > 0 for x in batch_negatives)

                if has_entity_neg:
                    # Build negative inputs by replacing tail tokens in the original text.
                    # We'll reuse tokenizer on modified strings. For efficiency this is kept small.
                    neg_texts = []
                    for i, sampled in enumerate(batch_negatives):
                        head = meta[i].get('head')
                        head_context = entity_incoming_neighbors.get(head, "")
                        for neg_tail in sampled:
                            tail_context = entity_incoming_neighbors.get(neg_tail, "")
                            neg_texts.append(f"{head} [SEP] {head_context} [SEP] {neg_tail} [SEP] {tail_context}")

                    if neg_texts:
                        neg_enc = tokenizer(neg_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=config.MAX_LENGTH)
                        neg_inputs = {k: v.to(rank, non_blocking=True) for k, v in neg_enc.items()}
                        with torch.no_grad():
                            neg_outputs = model.module(**neg_inputs) if hasattr(model, 'module') else model(**neg_inputs)
                        neg_logits = neg_outputs.logits  # (sum_k_batch, num_relations)

                        # Expand pos_scores to match negatives grouping
                        pos_scores = logits[torch.arange(logits.size(0)), labels]
                        # Compute hinge against negatives by grouping: simple approach — take mean neg score per sample
                        # Map neg_logits back to samples
                        per_sample_neg_means = []
                        idx = 0
                        for sampled in batch_negatives:
                            n = len(sampled)
                            if n > 0:
                                mean_neg = neg_logits[idx:idx+n].mean(dim=0)  # mean across neg tails
                                # take max negative relation score across classes
                                max_neg_val, _ = mean_neg.max(dim=0)
                                per_sample_neg_means.append(max_neg_val)
                                idx += n
                            else:
                                per_sample_neg_means.append(torch.tensor(-1e9, device=logits.device))

                        per_sample_neg = torch.stack(per_sample_neg_means).to(logits.device)
                        diffs = config.HNM_MARGIN - (pos_scores - per_sample_neg)
                        losses = torch.clamp(diffs, min=0.0)
                        hloss = config.HNM_WEIGHT * losses.mean()
                    else:
                        hloss = torch.tensor(0.0, device=logits.device)
                else:
                    hloss = hard_negative_hinge_loss(logits, labels,
                                                     top_k=config.HNM_TOP_K,
                                                     margin=config.HNM_MARGIN,
                                                     weight=config.HNM_WEIGHT)
            else:
                hloss = torch.tensor(0.0, device=logits.device)

            loss = loss_ce + hloss
            if loss.dim() > 0:
                loss = loss.mean()

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) == total_batches and (batch_idx + 1) % gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps

            if rank == 0 and (batch_idx + 1) % 100 == 0:
                current_loss = train_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{total_batches} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | Avg Loss: {current_loss:.4f}")

        if rank == 0:
            avg_train_loss = train_loss / total_batches
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Avg Train Loss: {avg_train_loss:.4f}")
            print(f"{'='*50}")

            print("Evaluating on validation set...")
            valid_results = evaluate_relation_model(
                model.module, valid_dataloader, rank,
                relation_to_idx, valid_triplets, tokenizer,
                entity_incoming_neighbors, max_length=config.MAX_LENGTH
            )

            print(f"  MR: {valid_results['MR']:.2f}")
            print(f"  MRR: {valid_results['MRR']:.4f}")
            print(f"  Hits@1: {valid_results['Hits@1']:.4f}")
            print(f"  Hits@3: {valid_results['Hits@3']:.4f}")
            print(f"  Hits@5: {valid_results['Hits@5']:.4f}")
            print(f"  Hits@10: {valid_results['Hits@10']:.4f}\n")

            save_test_results(epoch, valid_results, dataset_config["MODEL_SAVE_PATH"], task='relation_val')
            save_checkpoint(model.module, optimizer, epoch + 1,
                           os.path.join(dataset_config["MODEL_SAVE_PATH"], f'checkpoint_epoch_{epoch+1}.pth'))

            if valid_results['MRR'] > best_val_mrr:
                best_val_mrr = valid_results['MRR']
                epochs_without_improvement = 0
                save_checkpoint(model.module, optimizer, epoch + 1, best_checkpoint_path)
                print(f"Best validation MRR improved to {best_val_mrr:.4f}; saved checkpoint_best.pth")
            else:
                epochs_without_improvement += 1
                print(
                    f"No validation MRR improvement for {epochs_without_improvement} epoch(s) "
                    f"(patience={config.EARLY_STOPPING_PATIENCE})."
                )

            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {config.EARLY_STOPPING_PATIENCE} "
                    f"epoch(s) without validation improvement."
                )
                break

    if rank == 0:
        if os.path.exists(best_checkpoint_path):
            load_device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            best_checkpoint = torch.load(best_checkpoint_path, map_location=load_device)
            model.module.load_state_dict(best_checkpoint['model_state_dict'])

        print("\nTraining Completed Successfully for this dataset!")
        print("Evaluating best checkpoint on test set...")
        test_results = evaluate_relation_model(
            model.module, test_dataloader, rank,
            relation_to_idx, test_triplets, tokenizer,
            entity_incoming_neighbors, max_length=config.MAX_LENGTH
        )

        print(f"  MR: {test_results['MR']:.2f}")
        print(f"  MRR: {test_results['MRR']:.4f}")
        print(f"  Hits@1: {test_results['Hits@1']:.4f}")
        print(f"  Hits@3: {test_results['Hits@3']:.4f}")
        print(f"  Hits@5: {test_results['Hits@5']:.4f}")
        print(f"  Hits@10: {test_results['Hits@10']:.4f}\n")

        save_test_results(config.NUM_EPOCHS - 1, test_results, dataset_config["MODEL_SAVE_PATH"], task='relation')
        model.module.save_pretrained(dataset_config["MODEL_SAVE_PATH"])
        tokenizer.save_pretrained(dataset_config["MODEL_SAVE_PATH"])
        if os.path.exists(entity_neighbors_path):
            os.remove(entity_neighbors_path)

    cleanup()



# ==================== MAIN SCRIPT ====================
# This block loops through all datasets sequentially.
# For each dataset:
# - prints dataset info
# - spawns multiple processes (one per GPU) for DDP training

if __name__ == '__main__':
    print("=" * 100)
    print("KNOWLEDGE GRAPH COMPLETION – SEQUENTIAL MULTI-DATASET TRAINING")
    print("=" * 100)
    print(f"Total datasets: {len(config.DATASETS)}\n")

    for idx, dataset in enumerate(config.DATASETS):
        print(f"\n{'='*100}")
        print(f"DATASET {idx+1}/{len(config.DATASETS)} → {dataset['name'].upper()}")
        print(f"Model will be saved to: {dataset['MODEL_SAVE_PATH']}")
        print(f"{'='*100}\n")

        world_size = config.NUM_GPUS
        # Pass the full dataset config to every spawned process
        mp.spawn(train, args=(world_size, dataset), nprocs=world_size, join=True)

    print("\n ALL DATASETS PROCESSED SUCCESSFULLY! ")