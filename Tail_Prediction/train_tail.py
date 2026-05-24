import json
import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import config_tail

from dataset_tail import TailContextDataset
from evaluate_tail import evaluate_model
from utils_tail import load_json, save_json, save_checkpoint


def train_and_evaluate(
    model_name,
    tokenizer_class,
    model_class,
    processed_dir,
    output_dir,
    num_epochs,
    batch_size=16,
    learning_rate=5e-5,
    max_length=128,
    device="cpu",
):
    tokenizer = tokenizer_class.from_pretrained(model_name)

    tail_labels = load_json(os.path.join(processed_dir, "tail_label_vocab.json"))
    num_labels = len(tail_labels)

    model = model_class.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    train_dataset = TailContextDataset(
        os.path.join(processed_dir, "train_context.jsonl"),
        tokenizer,
        max_length=max_length,
    )
    valid_dataset = TailContextDataset(
        os.path.join(processed_dir, "valid_context.jsonl"),
        tokenizer,
        max_length=max_length,
    )

    # Deterministic DataLoader setup
    gen = torch.Generator()
    gen.manual_seed(config_tail.SEED)

    def worker_init_fn(worker_id):
        import random
        import numpy as _np
        import torch as _torch

        seed = config_tail.SEED + worker_id
        random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)

    # Auto-tune `num_workers` based on dataset size for balanced throughput
    dataset_size = len(train_dataset)
    cpu_count = os.cpu_count() or 1
    # Small datasets don't benefit from workers; medium/large do
    if dataset_size < 2000:
        num_workers = 0
    else:
        # one worker per ~2000 samples, capped by CPU count and an upper limit
        num_workers = min(max(1, dataset_size // 2000), cpu_count, 8)

    pin_memory = True if torch.cuda.is_available() else False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(config_tail.SEED),
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    best_mrr = -1.0
    validation_metrics_path = os.path.join(output_dir, "validation_metrics.jsonl")

    speed_metrics = {
        "train_time_per_epoch": [],
        "valid_time_per_epoch": [],
    }

    total_train_time = 0.0
    total_valid_time = 0.0

    for epoch in range(num_epochs):
        model.train()
        start_train = time.perf_counter()

        train_loss = 0.0
        for inputs, labels, _meta in train_dataloader:
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_train_time = time.perf_counter() - start_train
        total_train_time += epoch_train_time
        speed_metrics["train_time_per_epoch"].append(epoch_train_time)

        model.eval()
        start_valid = time.perf_counter()
        valid_metrics = evaluate_model(
            model,
            valid_dataloader,
            device,
            tail_labels,
            save_dir=None,
            split_name="valid",
        )
        epoch_valid_time = time.perf_counter() - start_valid
        total_valid_time += epoch_valid_time
        speed_metrics["valid_time_per_epoch"].append(epoch_valid_time)

        avg_train_loss = train_loss / max(len(train_dataloader), 1)

        log_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "valid_metrics": valid_metrics,
        }

        with open(validation_metrics_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(log_record) + "\n")

        if valid_metrics["MRR"] > best_mrr:
            best_mrr = valid_metrics["MRR"]
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

        checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)

    speed_metrics.update(
        {
            "total_training_time": total_train_time,
            "total_validation_time": total_valid_time,
            "train_triples_per_second": len(train_dataset) / total_train_time
            if total_train_time
            else 0.0,
            "valid_triples_per_second": len(valid_dataset) / total_valid_time
            if total_valid_time
            else 0.0,
        }
    )

    save_json(os.path.join(output_dir, "run_config.json"), {"best_valid_mrr": best_mrr})

    return {
        "best_model_dir": best_model_dir,
        "speed_metrics": speed_metrics,
        "tail_labels": tail_labels,
    }