import json
import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import config_tail

from dataset_tail import TailContextDataset
from evaluate_tail import evaluate_model
from utils_tail import (
    capture_rng_state,
    load_checkpoint,
    load_json,
    restore_rng_state,
    save_checkpoint,
    save_json,
)


def worker_init_fn(worker_id):
    # Top-level worker init so it's picklable on Windows
    import random
    import numpy as _np
    import torch as _torch
    import config_tail as _config_tail

    seed = _config_tail.SEED + worker_id
    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)


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
    resume_training=None,
    checkpoint_every_steps=None,
    max_train_seconds=None,
):
    if resume_training is None:
        resume_training = config_tail.RESUME_TRAINING
    if checkpoint_every_steps is None:
        checkpoint_every_steps = config_tail.CHECKPOINT_EVERY_STEPS
    if max_train_seconds is None and config_tail.MAX_TRAIN_SECONDS > 0:
        max_train_seconds = config_tail.MAX_TRAIN_SECONDS

    log_every_steps = max(1, config_tail.LOG_EVERY_STEPS)

    print(
        (
            "[train] Starting training "
            f"model={model_name}, epochs={num_epochs}, batch_size={batch_size}, "
            f"lr={learning_rate}, device={device}, seed={config_tail.SEED}"
        ),
        flush=True,
    )

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


    # For quick experiments on various platforms, use single-process dataloading
    # to avoid multiprocessing/pickling issues on Windows CI or constrained envs.
    num_workers = 0

    pin_memory = True if torch.cuda.is_available() else False

    print(
        (
            "[train] Data ready: "
            f"train_samples={len(train_dataset)}, valid_samples={len(valid_dataset)}, "
            f"num_workers={num_workers}, pin_memory={pin_memory}"
        ),
        flush=True,
    )

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
    latest_checkpoint_path = os.path.join(checkpoints_dir, "latest_checkpoint.pth")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    best_mrr = -1.0
    start_epoch = 0
    start_batch_index = 0
    global_step = 0
    validation_metrics_path = os.path.join(output_dir, "validation_metrics.jsonl")

    speed_metrics = {
        "train_time_per_epoch": [],
        "valid_time_per_epoch": [],
    }

    total_train_time = 0.0
    total_valid_time = 0.0

    if resume_training:
        print(f"[train] Resume enabled. Looking for checkpoint: {latest_checkpoint_path}", flush=True)
        checkpoint = load_checkpoint(model, optimizer, latest_checkpoint_path)
        if checkpoint:
            start_epoch = int(checkpoint.get("epoch", 0))
            start_batch_index = int(checkpoint.get("batch_index", 0))
            best_mrr = float(checkpoint.get("best_mrr", best_mrr))
            global_step = int(checkpoint.get("global_step", 0))
            total_train_time = float(checkpoint.get("total_train_time", 0.0))
            total_valid_time = float(checkpoint.get("total_valid_time", 0.0))
            restore_rng_state(checkpoint.get("rng_state"))
            print(
                (
                    "[train] Resuming from checkpoint: "
                    f"epoch={start_epoch + 1}, batch_index={start_batch_index}, "
                    f"global_step={global_step}, best_mrr={best_mrr:.4f}"
                ),
                flush=True,
            )
        else:
            print("[train] No checkpoint found. Starting fresh.", flush=True)

    if start_epoch >= num_epochs:
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
                "training_paused": False,
            }
        )
        save_json(os.path.join(output_dir, "run_config.json"), {"best_valid_mrr": best_mrr})
        return {
            "best_model_dir": best_model_dir,
            "speed_metrics": speed_metrics,
            "tail_labels": tail_labels,
            "training_paused": False,
        }

    training_paused = False
    runtime_start = time.perf_counter()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        start_train = time.perf_counter()

        epoch_seed = config_tail.SEED + epoch
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(epoch_seed)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=epoch_generator,
            worker_init_fn=worker_init_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        train_loss = 0.0
        total_batches = len(train_dataloader)
        skipped_notice_printed = False
        print(
            f"[train] Epoch {epoch + 1}/{num_epochs} started (batches={total_batches})",
            flush=True,
        )
        for batch_index, (inputs, labels, _meta) in enumerate(train_dataloader):
            if epoch == start_epoch and batch_index < start_batch_index:
                if not skipped_notice_printed and start_batch_index > 0:
                    print(
                        (
                            "[train] Skipping already-completed batches from resume: "
                            f"0..{start_batch_index - 1}"
                        ),
                        flush=True,
                    )
                    skipped_notice_printed = True
                continue

            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            global_step += 1

            if checkpoint_every_steps and global_step % checkpoint_every_steps == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    latest_checkpoint_path,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    best_mrr=best_mrr,
                    global_step=global_step,
                    total_train_time=total_train_time + (time.perf_counter() - start_train),
                    total_valid_time=total_valid_time,
                    rng_state=capture_rng_state(),
                )
                print(
                    f"[train] Checkpoint saved at global_step={global_step} -> {latest_checkpoint_path}",
                    flush=True,
                )

            current_batch = batch_index + 1
            if current_batch % log_every_steps == 0 or current_batch == total_batches:
                avg_loss_so_far = train_loss / max(1, current_batch - (start_batch_index if epoch == start_epoch else 0))
                pct = (current_batch / max(total_batches, 1)) * 100.0
                print(
                    (
                        f"[train] Epoch {epoch + 1}/{num_epochs} "
                        f"batch {current_batch}/{total_batches} ({pct:.1f}%) "
                        f"loss={avg_loss_so_far:.4f} global_step={global_step}"
                    ),
                    flush=True,
                )

            if max_train_seconds and (time.perf_counter() - runtime_start) >= max_train_seconds:
                training_paused = True
                save_checkpoint(
                    model,
                    optimizer,
                    latest_checkpoint_path,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    best_mrr=best_mrr,
                    global_step=global_step,
                    total_train_time=total_train_time + (time.perf_counter() - start_train),
                    total_valid_time=total_valid_time,
                    rng_state=capture_rng_state(),
                )
                print(
                    (
                        "[train] Time budget reached. "
                        f"Paused at epoch={epoch + 1}, batch={batch_index + 1}, global_step={global_step}"
                    ),
                    flush=True,
                )
                break

        if training_paused:
            break

        epoch_train_time = time.perf_counter() - start_train
        total_train_time += epoch_train_time
        speed_metrics["train_time_per_epoch"].append(epoch_train_time)
        start_batch_index = 0

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

        print(
            (
                f"[train] Epoch {epoch + 1}/{num_epochs} complete: "
                f"train_loss={avg_train_loss:.4f}, valid_MRR={valid_metrics['MRR']:.4f}, "
                f"Hits@1={valid_metrics['Hits@1']:.4f}, Hits@10={valid_metrics['Hits@10']:.4f}"
            ),
            flush=True,
        )

        if valid_metrics["MRR"] > best_mrr:
            best_mrr = valid_metrics["MRR"]
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"[train] New best model saved (best_MRR={best_mrr:.4f}) -> {best_model_dir}", flush=True)

        checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        save_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            epoch=epoch + 1,
            batch_index=0,
            best_mrr=best_mrr,
            global_step=global_step,
            total_train_time=total_train_time,
            total_valid_time=total_valid_time,
            rng_state=capture_rng_state(),
        )
        print(
            f"[train] Epoch checkpoint saved -> {checkpoint_path}",
            flush=True,
        )
        save_checkpoint(
            model,
            optimizer,
            latest_checkpoint_path,
            epoch=epoch + 1,
            batch_index=0,
            best_mrr=best_mrr,
            global_step=global_step,
            total_train_time=total_train_time,
            total_valid_time=total_valid_time,
            rng_state=capture_rng_state(),
        )

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
            "training_paused": training_paused,
        }
    )

    save_json(os.path.join(output_dir, "run_config.json"), {"best_valid_mrr": best_mrr})

    if training_paused:
        print("[train] Training paused with checkpoint ready for resume.", flush=True)
    else:
        print("[train] Training completed.", flush=True)

    return {
        "best_model_dir": best_model_dir,
        "speed_metrics": speed_metrics,
        "tail_labels": tail_labels,
        "training_paused": training_paused,
    }