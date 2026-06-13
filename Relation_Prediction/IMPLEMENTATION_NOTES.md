# General KG Relation Prediction Implementation Notes

## Scope

This version updates the relation-prediction implementation for **standard/general knowledge graph benchmarks**, not biomedical-only datasets. The main target datasets are:

- FB15k-237
- WN18RR
- CoDEx-S
- CoDEx-M
- YAGO3-10

FB15K and WN18 are intentionally removed from the main configuration because they are older, easier benchmarks with known inverse/duplicate relation leakage concerns. They can still be used as sanity checks, but not as primary paper evidence.

## Main method

The model remains strictly **negative-sample-free**.

```text
Input:  [HEAD_SCHEMA] schema_h [HEAD] h_text
        [HEAD_CTX] adaptive grouped context_h
        [TAIL_SCHEMA] schema_t [TAIL] t_text
        [TAIL_CTX] adaptive grouped context_t
        [PAIR_CTX] common-neighbor / relation-motif evidence
        optional [PATH] compressed 2-hop relation motifs

Loss:   CE / weighted CE / deferred weighted CE / focal loss

Output: softmax over train relation labels
```

No corrupted triples, no negative tails, no hard-negative mining, and no negative-forward pass are used.

## Implemented changes in this version

### 1. Benchmark/config updates

- Main dataset list now targets FB15k-237, WN18RR, CoDEx-S, CoDEx-M, and YAGO3-10.
- FB15K and WN18 are removed from the main config list.
- Optional `ENTITY_LABEL_FILE` and `RELATION_LABEL_FILE` fields are supported per dataset.
- Default `USE_ENTITY_TYPES=False` because generic KG benchmarks often lack clean type files.
- Default `USE_SCHEMA_TOKENS=True` and `USE_SCHEMA_PRIOR=True` for induced schema-aware relation prediction.

### 2. Generic KG text normalization

Added:

```python
normalize_kg_text(raw, dataset_name, label_map)
load_label_map(label_file)
```

This converts raw KG IDs/URIs/synsets into more readable encoder text, for example:

```text
/people/person/place_of_birth -> people person place of birth
__good_a_01                  -> good a 01
<http://yago/entity_name>     -> yago entity name
```

If label-map files are provided, they override heuristic normalization.

### 3. Dynamic padding

- Added `DYNAMIC_PADDING=True`.
- Default `PRETOKENIZE=False`.
- Added `make_relation_collate_fn()` to dynamically pad each batch instead of padding every sample to `MAX_LENGTH`.

This should improve speed and memory usage without changing the model.

### 4. Pair-specific context

Added `[PAIR_CTX]` evidence for each `(head, ?, tail)` query:

- common neighbors,
- compressed 2-hop relation motifs,
- known train-pair relations if the same `(head, tail)` pair has other train relations.

This is now enabled by default:

```python
USE_PAIR_CONTEXT = True
MAX_PAIR_CONTEXT_TOKENS = 24
```

### 5. Compressed path motifs

Path context now supports:

```python
PATH_CONTEXT_MODE = "motif"  # motif | full
```

Motif mode writes only relation patterns:

```text
[PATH] out:born in -> out:located in
```

Full mode writes entity paths and is kept as an ablation because it costs more tokens.

### 6. Grouped/compressed context

Added:

```python
GROUP_CONTEXT_BY_RELATION = True
MAX_NEIGHBORS_PER_RELATION = 3
```

Instead of repeated items:

```text
[H_OUT] contains Paris
[H_OUT] contains Lyon
[H_OUT] contains Marseille
```

it writes:

```text
[H_OUT] contains: Paris | Lyon | Marseille
```

This reduces repeated relation tokens and improves context/token efficiency.

### 7. Induced schema signatures

For each entity, the code now creates a train-only schema signature based on:

- degree bucket,
- most common outgoing relations,
- most common incoming relations.

Example:

```text
d4_10|O:located_in,member_of|I:born_in
```

The input can include:

```text
[HEAD_SCHEMA] ...
[TAIL_SCHEMA] ...
```

### 8. Soft schema prior

The code builds a train-only prior:

```text
P(relation | head_schema, tail_schema)
```

and softly adjusts logits:

```python
logits = logits + lambda * log_prior
```

This is **not** a hard mask. It never blocks a relation. It is safer than type masking for incomplete general KGs.

### 9. Filtered relation-ranking evaluation

Added filtered relation ranking:

```text
Raw MRR/Hits:      gold relation ranked against all relation labels
Filtered MRR/Hits: other known true relations for same (head, tail) are masked
```

This is important when multiple relations can connect the same entity pair.

### 10. Positive-only long-tail handling

Added or extended:

- `LOSS_TYPE = "deferred_weighted_ce"`
- `REWEIGHT_START_EPOCH`
- `TRAIN_SAMPLER = "relation_balanced"`
- relation frequency buckets in evaluation.

This improves rare-relation exposure without using negative sampling.

### 11. Better checkpoint selection

Added:

```python
CHECKPOINT_METRIC = "mrr_macro_f1"
```

This selects checkpoints using:

```text
0.7 * MRR + 0.3 * Macro-F1
```

when rare relation quality matters.

## Not implemented yet

The following are still future extensions:

1. Relation-prototype classifier fusion.
2. Graph-feature fusion MLP after CLS.
3. Teacher-student context distillation.
4. Learned context policy.
5. Full per-relation confusion matrix export.
6. Multi-hop paths beyond 2-hop.

## Suggested experiment order

Run ablations in this order:

| Run | Configuration | Goal |
|---:|---|---|
| A | Current baseline, no pair/path/schema prior | Clean baseline |
| B | + text normalization | Improve encoder signal |
| C | + dynamic padding | Speed/memory improvement |
| D | + grouped context | Lower token cost |
| E | + pair context | Accuracy improvement |
| F | + motif path context | Optional path gain |
| G | + soft schema prior | Hits@1/MRR gain |
| H | + deferred weighted CE | Rare relation improvement |
| I | + relation-balanced sampler | Rare relation improvement |

## Validation performed in sandbox

- `python -m py_compile` passed for the updated Python files.
- A synthetic data-loader test verified:
  - target-triple relation leakage is excluded from head/tail context,
  - pair context is generated,
  - dynamic padding collate works,
  - filtered relation mask works.

Full training was not run because the real datasets and pretrained model cache are not available in the sandbox.
