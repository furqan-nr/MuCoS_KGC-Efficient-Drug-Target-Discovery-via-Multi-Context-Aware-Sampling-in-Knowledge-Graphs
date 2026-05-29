Improvements
============

This file lists code changes made to this repository and the practical effects they are expected to have. Statements are descriptive (what changed) and conservative about effects (what to expect), not promotional claims.

- Pre-tokenized caches
  - What changed: preprocessing now saves tokenized inputs to `processed/*_tokenized.pt` and the dataset can load these directly (see `Tail_Prediction/preprocess_contexts.py` and `Tail_Prediction/dataset_tail.py`).
  - Expected effect: training and evaluation can read pre-built tensors instead of re-tokenizing on the fly, which reduces CPU work and should increase throughput and make timing measurements more reproducible.

- Dynamic padding (pad-to-multiple-of)
  - What changed: the collate function uses `pad_to_multiple_of` when building batches (`Tail_Prediction/utils_tail.py`).
  - Expected effect: batch tensors align to sizes that are friendlier for GPU kernels, which commonly improves GPU utilization and can increase effective batch throughput.

- Mixed precision (AMP)
  - What changed: training code is wired to use AMP when available and enabled (`Tail_Prediction/train_tail.py`, controlled by `MUCOS_USE_AMP`).
  - Expected effect: reduced GPU memory use and typically faster training on CUDA devices while maintaining comparable model quality when used correctly.

- Faster, memory-efficient evaluation
  - What changed: evaluation uses logits, `topk`, and on-device application of priors/constraints instead of full CPU-side softmax/sorts (`Tail_Prediction/evaluate_tail.py`).
  - Expected effect: fewer CPU↔GPU transfers and less memory/sorting work on the host, which should reduce evaluation latency and increase triples/sec processed during evaluation.

- Relation / type-constrained candidate sets
  - What changed: relation→tail candidate lists are precomputed and saved to `processed/relation_tail_candidates.json` and optionally used at evaluation (`Tail_Prediction/preprocess_contexts.py`, `Tail_Prediction/main_tail.py`).
  - Expected effect: when constraints are valid, the candidate set is smaller which reduces ranking work and can improve ranking metrics by ignoring implausible tails.

- Truncation-aware context ordering
  - What changed: new `MUCOS_CONTEXT_ORDER=auto` option in preprocessing keeps the relation context ordering when the default ordering would cause the relation to be truncated (`Tail_Prediction/preprocess_contexts.py`).
  - Expected effect: preserves important relation information for cases with tight token budgets, which can improve accuracy for long-context examples.

- Enforce cache-for-speed option
  - What changed: `MUCOS_REQUIRE_TOKENIZED_CACHE` enforces that datasets use pre-built tokenized caches and fail if they are missing (`Tail_Prediction/config_tail.py`, `Tail_Prediction/dataset_tail.py`).
  - Expected effect: ensures runs that measure speed use the same fast path (no hidden fallback to on-the-fly tokenization), making speed comparisons reproducible. Note: preprocessing must be run first to produce the caches.

- Small correctness fix
  - What changed: fixed AMP scaler initialization bug in `Tail_Prediction/train_tail.py` so mixed-precision runs use the correct GradScaler API.
  - Expected effect: prevents a runtime error and allows AMP-enabled training to run as intended.


Notes and caveats

- The repository changes described above reduce CPU work and aim to improve throughput and evaluation latency, but measured gains depend on hardware (CPU, GPU), dataset size, and runtime flags. Please run the pipeline with your target hardware to confirm improvements.


- Type compatibility implemented
  - What changed: the sampling helper `_type_compatibility()` is implemented in `Tail_Prediction/sampling.py`. It supports relation->tail sets and entity->type heuristics and returns a conservative compatibility score used during candidate scoring.
  - Expected effect: when type or relation-candidate information is available, sampling can prefer more plausible tails, which improves the quality of selected contexts without overpowering other heuristics.