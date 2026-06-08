**Improvements (Tail Prediction only)**

This document lists the practical improvements implemented for the Tail Prediction pipeline (`Tail_Prediction/`). Each item describes the change and the expected effect.

- **Pre-tokenized caches:** Preprocessing saves tokenized inputs to `processed/*_tokenized.pt` and the dataset loader can load these tensors directly (`Tail_Prediction/preprocess_contexts.py`, `Tail_Prediction/dataset_tail.py`). Expected effect: large reduction in CPU tokenization overhead and more reproducible timing.

- **Dynamic padding (pad-to-multiple-of):** Collate uses `pad_to_multiple_of` to align batch tensor widths (`Tail_Prediction/utils_tail.py`). Expected effect: better GPU kernel utilization and potentially higher throughput.

- **Mixed precision (AMP):** Training is wired to use AMP when enabled via `MUCOS_USE_AMP` (`Tail_Prediction/train_tail.py`). Expected effect: lower memory footprint and faster training on supported GPUs.

- **Faster, memory-efficient evaluation:** Evaluation uses on-device `topk` and logits-based ranking to minimize CPU↔GPU transfers and host-side sorting (`Tail_Prediction/evaluate_tail.py`). Expected effect: reduced evaluation latency and higher triples/sec.

- **Relation / type-constrained candidate sets:** Precompute relation→tail candidate lists (saved under `processed/`) and apply them at evaluation when enabled (`Tail_Prediction/preprocess_contexts.py`, `Tail_Prediction/main_tail.py`). Expected effect: smaller candidate sets for ranking, faster evaluation, and improved ranking accuracy when constraints are valid.

- **Truncation-aware context ordering:** New `MUCOS_CONTEXT_ORDER=auto` option preserves relation/context ordering when truncation would otherwise remove the relation token (`Tail_Prediction/preprocess_contexts.py`). Expected effect: better accuracy on long-context examples.

- **Enforce cache-for-speed option:** `MUCOS_REQUIRE_TOKENIZED_CACHE` ensures runs that aim to measure speed use pre-built tokenized caches instead of silently falling back to on-the-fly tokenization (`Tail_Prediction/config_tail.py`, `Tail_Prediction/dataset_tail.py`). Expected effect: reproducible speed measurements; requires running preprocessing first.

- **Type-compatibility sampling helper:** `_type_compatibility()` implemented in `Tail_Prediction/sampling.py` to bias sampling towards type-compatible tails and relations. Expected effect: improved sampling quality and context relevance when type information exists.

- **Small correctness fixes:** Fixed GradScaler initialization and related AMP issues in `Tail_Prediction/train_tail.py` to avoid runtime errors when AMP is enabled.

Notes and caveats

- These changes focus on CPU/GPU throughput, evaluation latency, and stability for the Tail Prediction pipeline. Measured improvements depend on dataset size and hardware. To reproduce speed-sensitive runs, enable tokenized caches and use the same `MUCOS_*` runtime flags across experiments.


**Relation Prediction Improvements**

- **Training-only contexts / no leakage:** Contexts and neighbor precomputation are built from training triples only (`Relation_Prediction/train.py`, `Relation_Prediction/data_loader.py`). Expected effect: eliminates validation/test leakage and produces honest validation metrics.

- **Precompute and cache entity neighbors:** `precompute_entity_info()` writes dataset-local neighbor caches (saved under each `MODEL_SAVE_PATH`) and other ranks load the cache to avoid redundant computation. Expected effect: faster multi-process startup and consistent neighbor data across ranks.

- **Tokenization caching & batched evaluation:** Relation datasets pre-tokenize inputs and evaluation uses batched `topk` ranking on-device (`Relation_Prediction/data_loader.py`, `Relation_Prediction/utils.py`). Expected effect: reduced CPU tokenization and faster evaluation with fewer CPU↔GPU transfers.

- **DDP-ready training loop & checkpoints:** `Relation_Prediction/train.py` uses DistributedDataParallel, gradient accumulation, validation-based checkpointing and early stopping. Expected effect: scalable multi-GPU training with reproducible checkpoints and automatic early stop on no improvement.

- **Hard-negative mining (HNM) basic integration:** Config flags and helper `hard_negative_hinge_loss` added; hinge loss combined with cross-entropy in the training loop when enabled (`Relation_Prediction/config.py`, `Relation_Prediction/utils.py`, `Relation_Prediction/train.py`). Expected effect: low-complexity emphasis on hard negatives to improve ranking metrics with minimal model changes.


**Specific Relation Improvements**

- **Removed drug-target-only domain filter:** `Specific_R_Prediction` no longer hard-filters to drug-target relations by default (`Specific_R_Prediction/utils.py`, `Specific_R_Prediction/main.py`). Expected effect: dataset becomes benchmark-ready for general KGC evaluation.

- **Tokenization caching:** `Specific_R_Prediction/dataset.py` pre-tokenizes and caches encodings for faster training and evaluation. Expected effect: lower CPU overhead and consistent batch construction.

- **Batched evaluation and ranking helpers:** `_rank_batch` and batched evaluation utilities return ranks and metrics using on-device operations (`Specific_R_Prediction/utils.py`). Expected effect: faster validation/test runs and lower memory overhead.

- **Hard-negative mining integration:** `hard_negative_hinge_loss` added and integrated into `Specific_R_Prediction/train.py` training loop as a CE + hinge combination. Expected effect: similar to relation pipeline — emphasis on hard negatives with low engineering cost.

- **General correctness & stability fixes:** small fixes across `Specific_R_Prediction` (tokenization, label mapping, error-handling in training loop) to improve robustness during experiments.

Summary

- These additions consolidate improvements across `Tail_Prediction/`, `Relation_Prediction/` and `Specific_R_Prediction/` focusing on speed, reproducibility, and a lightweight boost to ranking accuracy via hard-negative emphasis. For reproducible speed measurements enable pre-tokenized caches and use the same `MUCOS_*` runtime flags.
