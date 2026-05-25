import os
import time

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

import config_tail
from evaluate_tail import evaluate_model
from preprocess_contexts import preprocess_all
from reproducibility import save_json, set_seed
from train_tail import train_and_evaluate
from utils_tail import load_json
from dataset_tail import TailContextDataset
from torch.utils.data import DataLoader


def _processed_outputs_exist():
    required_files = [
        os.path.join(config_tail.PROCESSED_DIR, "train_context.jsonl"),
        os.path.join(config_tail.PROCESSED_DIR, "valid_context.jsonl"),
        os.path.join(config_tail.PROCESSED_DIR, "test_context.jsonl"),
        os.path.join(config_tail.PROCESSED_DIR, "dataset_stats.json"),
        os.path.join(config_tail.PROCESSED_DIR, "tail_label_vocab.json"),
    ]
    return all(os.path.exists(path) for path in required_files)


def main():
    print("[pipeline] Run started", flush=True)
    set_seed(config_tail.SEED)
    print(f"[pipeline] Seed set to {config_tail.SEED}", flush=True)
    print(
        (
            "[pipeline] Paths: "
            f"train={config_tail.train_file_path}, valid={config_tail.valid_file_path}, "
            f"test={config_tail.test_file_path}, processed={config_tail.PROCESSED_DIR}, "
            f"outputs={config_tail.OUTPUT_DIR}"
        ),
        flush=True,
    )

    preprocess_time = 0.0
    if _processed_outputs_exist():
        print(
            f"[pipeline] Processed data already exists in {config_tail.PROCESSED_DIR}; skipping preprocessing.",
            flush=True,
        )
    else:
        preprocess_start = time.perf_counter()
        print("[pipeline] Preprocessing started", flush=True)
        preprocess_all()
        preprocess_time = time.perf_counter() - preprocess_start
        print(f"[pipeline] Preprocessing finished in {preprocess_time:.2f}s", flush=True)

    os.makedirs(config_tail.OUTPUT_DIR, exist_ok=True)

    print("[pipeline] Training started", flush=True)
    train_results = train_and_evaluate(
        model_name=config_tail.MODEL_NAME,
        tokenizer_class=DistilBertTokenizer,
        model_class=DistilBertForSequenceClassification,
        processed_dir=config_tail.PROCESSED_DIR,
        output_dir=config_tail.OUTPUT_DIR,
        num_epochs=config_tail.NUM_EPOCHS,
        batch_size=config_tail.BATCH_SIZE,
        learning_rate=config_tail.LEARNING_RATE,
        max_length=config_tail.MAX_LENGTH,
        device=config_tail.device,
    )
    print("[pipeline] Training phase finished", flush=True)

    if train_results.get("training_paused"):
        save_json(
            os.path.join(config_tail.OUTPUT_DIR, "run_status.json"),
            {"status": "paused", "reason": "time_budget_or_manual_checkpoint", "seed": config_tail.SEED},
        )
        print("[pipeline] Run paused. Resume by re-running the same command.", flush=True)
        return

    print("[pipeline] Test evaluation started", flush=True)
    test_dataset = TailContextDataset(
        os.path.join(config_tail.PROCESSED_DIR, "test_context.jsonl"),
        DistilBertTokenizer.from_pretrained(config_tail.MODEL_NAME),
        max_length=config_tail.MAX_LENGTH,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=config_tail.BATCH_SIZE, shuffle=False)

    best_model_dir = train_results["best_model_dir"]
    model = DistilBertForSequenceClassification.from_pretrained(best_model_dir)
    model.to(config_tail.device)

    test_start = time.perf_counter()
    test_metrics = evaluate_model(
        model,
        test_dataloader,
        config_tail.device,
        train_results["tail_labels"],
        save_dir=config_tail.OUTPUT_DIR,
        split_name="test",
    )
    test_time = time.perf_counter() - test_start
    print(f"[pipeline] Test evaluation finished in {test_time:.2f}s", flush=True)

    save_json(os.path.join(config_tail.OUTPUT_DIR, "test_metrics.json"), test_metrics)
    print(f"[pipeline] Test metrics saved -> {os.path.join(config_tail.OUTPUT_DIR, 'test_metrics.json')}", flush=True)

    speed_metrics = train_results["speed_metrics"]
    speed_metrics.update(
        {
            "context_preprocessing_time": preprocess_time,
            "test_evaluation_time": test_time,
            "test_triples_per_second": len(test_dataset) / test_time if test_time else 0.0,
        }
    )

    if torch.cuda.is_available():
        speed_metrics["gpu_max_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    save_json(os.path.join(config_tail.OUTPUT_DIR, "speed_metrics.json"), speed_metrics)

    run_config = {
        "method": "Reproducible Budgeted MuCoS Tail Prediction",
        "model": config_tail.MODEL_NAME,
        "max_length": config_tail.MAX_LENGTH,
        "max_hc": config_tail.MAX_HC,
        "max_rc": config_tail.MAX_RC,
        "max_total_context": config_tail.MAX_TOTAL_CONTEXT,
        "max_rc_if_hc_short": config_tail.MAX_RC_IF_HC_SHORT,
        "max_same_relation_in_hc": config_tail.MAX_SAME_RELATION_IN_HC,
        "sampling": "density + relation-aware + diversity-aware",
        "context_graph": "train_only",
        "exclude_current_training_triple_from_context": True,
        "loss": "cross_entropy",
        "negative_sampling": False,
        "seed": config_tail.SEED,
    }
    save_json(os.path.join(config_tail.OUTPUT_DIR, "run_config.json"), run_config)

    dataset_stats = load_json(os.path.join(config_tail.PROCESSED_DIR, "dataset_stats.json"))
    save_json(os.path.join(config_tail.OUTPUT_DIR, "dataset_stats.json"), dataset_stats)
    print("[pipeline] Run completed successfully", flush=True)


if __name__ == "__main__":
    main()