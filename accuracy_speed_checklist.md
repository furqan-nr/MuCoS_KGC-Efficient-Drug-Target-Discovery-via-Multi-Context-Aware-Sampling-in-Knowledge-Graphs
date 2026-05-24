# Accuracy and Scalability Checklist

Purpose: pinpoint exact code locations where small adjustments can improve accuracy without hurting speed or scalability. This is a planning checklist only.

## Tail Prediction (MuCo-KGC / MuCoS tail task)

- Align train/eval inputs so relation context is always used.
  - Location: [Tail_Prediction/train_tail.py](Tail_Prediction/train_tail.py#L1-L200)
  - Focus: `evaluate_model()` builds input as "head + head_context + relation" but omits relation context; training uses head+Hc+relation+Rc in `KGDataset`.

- Cache head and relation contexts so they are computed once.
  - Location: [Tail_Prediction/dataset_tail.py](Tail_Prediction/dataset_tail.py#L1-L120)
  - Focus: `get_one_hop_head_entity_neighbors()` and `get_relation_neighbors()` called inside `__getitem__` for every sample.

- Make input length configurable and consistent.
  - Location: [Tail_Prediction/dataset_tail.py](Tail_Prediction/dataset_tail.py#L1-L120)
  - Focus: `max_length=128` is hardcoded; align with config value used in `main_tail.py`.

- Use tokenizer-aware separators instead of literal "[SEP]" strings.
  - Location: [Tail_Prediction/dataset_tail.py](Tail_Prediction/dataset_tail.py#L1-L120)
  - Focus: string formatting for input sequence uses literal tokens.

## Relation Prediction (MuCoS relation task)

- Align train/eval inputs so contexts are used in evaluation.
  - Location: [Specific_R_Prediction/train.py](Specific_R_Prediction/train.py#L1-L200)
  - Focus: `evaluate_model()` uses only "head [SEP] tail", while training uses head+Hc+tail+Tc in `KGDataset`.

- Cache head and tail contexts so they are computed once.
  - Location: [Specific_R_Prediction/utils.py](Specific_R_Prediction/utils.py#L1-L200)
  - Focus: `get_one_hop_head_entity_neighbors()` and `get_one_hop_tail_entity_neighbors()` are called in `__getitem__` for every sample.

- Make input length configurable and consistent.
  - Location: [Specific_R_Prediction/dataset.py](Specific_R_Prediction/dataset.py#L1-L120)
  - Focus: `max_length=128` is hardcoded.

- Use tokenizer-aware separators instead of literal "[SEP]" strings.
  - Location: [Specific_R_Prediction/dataset.py](Specific_R_Prediction/dataset.py#L1-L120)
  - Focus: string formatting for input sequence uses literal tokens.

## Shared / MuCoS-Specific

- Enforce consistent density-based sampling across all contexts.
  - Locations:
    - [Tail_Prediction/dataset_tail.py](Tail_Prediction/dataset_tail.py#L1-L120)
    - [Specific_R_Prediction/utils.py](Specific_R_Prediction/utils.py#L1-L200)
  - Focus: sampling currently uses degree-based sorting; MuCoS paper defines density-based selection. Keeping thresholds small preserves speed.

- Precompute context dictionaries once per dataset split.
  - Locations:
    - [Tail_Prediction/main_tail.py](Tail_Prediction/main_tail.py#L1-L120)
    - [Specific_R_Prediction/train.py](Specific_R_Prediction/train.py#L1-L200)
  - Focus: build and reuse context maps keyed by entity/relation to reduce per-batch CPU overhead.
