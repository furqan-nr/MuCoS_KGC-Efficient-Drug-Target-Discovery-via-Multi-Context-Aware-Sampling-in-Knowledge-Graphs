# General KG Relation Prediction Implementation Notes — Updated

## Scope

This codebase implements a **negative-sample-free relation prediction** pipeline for standard/general knowledge graph benchmarks.

The configured benchmark list is:

- FB15k-237
- WN18RR
- CoDEx-S
- CoDEx-M
- YAGO3-10

The task is relation prediction:

```text
Input:  (head, ?, tail)
Output: relation label r
```

The relation label space is built from **training relations only**. Validation and test splits are checked to ensure that their relation labels are present in the training label space.

FB15K and WN18 are intentionally excluded from the main benchmark list because they are older datasets with known inverse/duplicate-relation leakage concerns. They may be used as sanity checks but should not be treated as primary evidence.

---

## Current implementation status

The implementation is mostly consistent with the previous notes, but the following corrections are important:

1. The standalone `config.py` default loss is:

```python
LOSS_TYPE = "ce"
```

not `deferred_weighted_ce`.

2. The standalone `config.py` default training sampler is:

```python
TRAIN_SAMPLER = "random"
```

not `relation_balanced`.

3. The Kaggle notebook overrides some runtime settings and uses:

```python
LOSS_TYPE = "deferred_weighted_ce"
TRAIN_SAMPLER = "relation_balanced"
PRETOKENIZE = True
```

So the notes should distinguish between **standalone Python defaults** and **notebook runtime overrides**.

4. Pair context and path context are different:
   - `[PAIR_CTX]` is enabled by default and may include common neighbors and compressed two-hop relation motifs.
   - Standalone `[PATH]` context is implemented but disabled by default with `USE_PATH_CONTEXT = False`.

5. Per-label F1 statistics are computed internally, but the current evaluation return object and saved metric files do not export the full per-label table or confusion matrix. Full confusion-matrix export remains future work.

---

## Main method

The model remains strictly **negative-sample-free**.

```text
Input:  [HEAD_SCHEMA] schema_h
        [HEAD] h_text
        [HEAD_CTX] adaptive grouped context_h
        [TAIL_SCHEMA] schema_t
        [TAIL] t_text
        [TAIL_CTX] adaptive grouped context_t
        [PAIR_CTX] common-neighbor / two-hop motif / other-pair-relation evidence
        optional [PATH] compressed two-hop path motifs

Loss:   CE by default
        optional weighted CE / deferred weighted CE / focal loss

Output: softmax over training relation labels
```

No corrupted triples, no negative tails, no hard-negative mining, and no negative-forward pass are used.

The model predicts the relation label directly from the encoded head-tail-context sequence.

---

## Default standalone configuration

The current standalone Python configuration uses:

```python
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 64
NUM_EPOCHS = 20
GRADIENT_ACCUMULATION_STEPS = 2
PRETOKENIZE = False
DYNAMIC_PADDING = True

USE_PAIR_CONTEXT = True
USE_PATH_CONTEXT = False
USE_SECOND_HOP_CONTEXT = False

USE_SCHEMA_TOKENS = True
USE_SCHEMA_PRIOR = True
SCHEMA_PRIOR_WEIGHT = 0.10

LOSS_TYPE = "ce"
TRAIN_SAMPLER = "random"

CHECKPOINT_METRIC = "mrr_macro_f1"
FILTERED_RELATION_EVAL = True
```

The effective batch size in standalone mode is:

```text
BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS = 64 × 2 = 128
```

---

## Kaggle notebook runtime overrides

The notebook version is a Kaggle-oriented execution wrapper. It patches the runtime configuration after importing `config.py`.

The notebook uses:

```python
NUM_EPOCHS = 20
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
EARLY_STOPPING_PATIENCE = 10
SAVE_EVERY_EPOCH = True

LOSS_TYPE = "deferred_weighted_ce"
REWEIGHT_START_EPOCH = 3
TRAIN_SAMPLER = "relation_balanced"
RELATION_SAMPLER_POWER = 0.5

PRETOKENIZE = True
DYNAMIC_PADDING = True
FILTERED_RELATION_EVAL = True
```

The notebook also overrides dataset paths for Kaggle, copies available checkpoint files into the output folder, and uses a custom AMP training loop.

Therefore, manuscript or README descriptions should specify which execution mode produced the reported results:

- standalone `train.py`, or
- Kaggle notebook run with runtime overrides.

---

## Implemented components

### 1. Benchmark/config setup

The main dataset list targets:

- FB15k-237
- WN18RR
- CoDEx-S
- CoDEx-M
- YAGO3-10

Each dataset entry contains:

```python
TRAIN_FILE_PATH
VALID_FILE_PATH
TEST_FILE_PATH
MODEL_SAVE_PATH
ENTITY_TYPE_FILE
ENTITY_LABEL_FILE
RELATION_LABEL_FILE
```

Entity and relation label maps are optional TSV files:

```text
raw_id<TAB>readable_label
```

They are used only for encoder text. Raw IDs are preserved internally for graph indexing and evaluation.

---

### 2. Generic KG text normalization

The following functions are implemented:

```python
normalize_kg_text(raw, dataset_name, label_map)
load_label_map(label_file)
```

They convert raw KG IDs, URIs, relation paths, and WordNet-style synsets into more readable encoder text.

Examples:

```text
/people/person/place_of_birth -> people person place of birth
__good_a_01                  -> good a 01
<http://yago/entity_name>     -> yago entity name
```

If a label-map file is provided, the label-map value overrides heuristic normalization.

---

### 3. Train-only graph precomputation

Graph statistics are computed from the training triples only. Validation and test triples are not used for graph context construction.

The graph cache stores:

```python
entity_degrees
out_degrees
in_degrees
relation_counts
total_relation_mentions
outgoing_map
incoming_map
undirected_map
type_pair_to_relations
pair_to_relations
schema_pair_relation_counts
entity_schema
triples_set
entities
```

This supports leakage-aware context construction while still allowing filtered evaluation to use all known true relations separately.

---

### 4. Relation label space

The classifier label space is built from training relations only:

```python
relation_to_idx = build_relation_to_idx(train_triplets)
```

Validation and test relations are checked with:

```python
validate_relation_coverage(...)
```

If validation or test contains unseen relation labels, the code raises an error.

---

### 5. Adaptive entity context

For each entity, the code builds role-aware head/tail context using outgoing and incoming training edges.

The context budget is adaptive:

```text
degree <= 2      -> max_context
degree <= 10     -> base_context
high-degree hub  -> reduced budget
```

This gives sparse nodes more context and compresses hubs.

Neighbor ranking uses:

```text
score = degree_score
        + relation_rarity_score
        - redundancy_penalty
```

where:

```text
degree_score       = log(1 + degree(neighbor))
relation_rarity    = log(1 + total_relation_mentions / (1 + relation_count))
redundancy_penalty = log(1 + relation_frequency_inside_context)
```

This keeps density central but also gives rare relations some priority and reduces repeated relation evidence.

---

### 6. Target-triple exclusion

When building context for a training/evaluation triple `(h, r, t)`, the target edge is excluded from head/tail context.

This prevents direct leakage of the gold relation into the input sequence.

For example, when constructing head context for `(h, r, t)`, the edge:

```text
h --r--> t
```

is removed from the context.

---

### 7. Grouped/compressed context

Grouped context is enabled by default:

```python
GROUP_CONTEXT_BY_RELATION = True
MAX_NEIGHBORS_PER_RELATION = 3
```

Instead of writing repeated items:

```text
[H_OUT] contains Paris
[H_OUT] contains Lyon
[H_OUT] contains Marseille
```

the code writes:

```text
[H_OUT] contains: Paris | Lyon | Marseille
```

This reduces repeated relation tokens and improves token efficiency.

---

### 8. Pair-specific context

Pair context is enabled by default:

```python
USE_PAIR_CONTEXT = True
MAX_COMMON_NEIGHBORS = 5
MAX_PAIR_MOTIFS = 5
MAX_PAIR_CONTEXT_TOKENS = 24
```

For each `(head, ?, tail)` query, the code may add:

1. common neighbors,
2. compressed two-hop relation motifs,
3. other known training relations for the same `(head, tail)` pair, excluding the current target relation.

The pair-context block is written with `[PAIR_CTX]`.

Important distinction:

```text
[PAIR_CTX] motifs are enabled through USE_PAIR_CONTEXT.
[PATH] items are controlled separately by USE_PATH_CONTEXT.
```

---

### 9. Optional path context

Standalone path context is implemented but disabled by default:

```python
USE_PATH_CONTEXT = False
PATH_CONTEXT_MODE = "motif"
MAX_PATHS = 3
```

If enabled, it adds `[PATH]` items.

Two modes are supported:

```python
PATH_CONTEXT_MODE = "motif"  # relation-only pattern
PATH_CONTEXT_MODE = "full"   # includes middle entity text
```

Motif mode writes compressed relation patterns such as:

```text
[PATH] out:born in -> out:located in
```

Full mode writes entity-level paths and is more token-expensive.

---

### 10. Optional second-hop local expansion

Second-hop local entity expansion is implemented but disabled by default:

```python
USE_SECOND_HOP_CONTEXT = False
SECOND_HOP_BUDGET = 5
```

If enabled, the entity context builder adds a small number of `[2HOP]` items from neighbors of selected neighbors.

---

### 11. Induced schema signatures

Explicit entity types are off by default:

```python
USE_ENTITY_TYPES = False
```

Instead, the code induces schema signatures from training-only graph structure.

Each entity schema contains:

1. a degree bucket,
2. top outgoing relations,
3. top incoming relations.

Example:

```text
d4_10|O:located_in,member_of|I:born_in
```

Schema tokens are enabled by default:

```python
USE_SCHEMA_TOKENS = True
```

The input sequence can therefore include:

```text
[HEAD_SCHEMA] ...
[TAIL_SCHEMA] ...
```

---

### 12. Soft schema prior

The code builds a train-only prior:

```text
P(relation | head_schema, tail_schema)
```

and applies it as a soft logit adjustment:

```python
logits = logits + lambda * log_prior
```

Default:

```python
USE_SCHEMA_PRIOR = True
SCHEMA_PRIOR_WEIGHT = 0.10
SCHEMA_PRIOR_SMOOTHING = 1.0
```

This is not a hard mask. It never blocks a relation. It only nudges logits toward relations frequently observed for the induced schema pair.

---

### 13. Model loader

The model is loaded through HuggingFace auto classes:

```python
AutoTokenizer
AutoModelForSequenceClassification
```

This makes it possible to use DistilBERT, BERT, RoBERTa, DeBERTa, or another encoder-only classifier without changing the training script.

The default model is:

```python
distilbert-base-uncased
```

Additional special tokens are added to the tokenizer, and the model token embeddings are resized accordingly.

---

### 14. Training objective

The standalone default loss is standard cross-entropy:

```python
LOSS_TYPE = "ce"
```

The following losses are implemented:

```python
ce
weighted_ce
deferred_weighted_ce
focal
```

Class weights are computed only when using:

```python
weighted_ce
deferred_weighted_ce
focal
```

For `deferred_weighted_ce`, the code uses standard CE before `REWEIGHT_START_EPOCH`, then switches to weighted CE.

The method remains positive-label-only in all cases.

---

### 15. Relation-balanced sampler

A positive-only relation-balanced sampler is implemented:

```python
TRAIN_SAMPLER = "relation_balanced"
```

However, it is not the standalone default. The standalone default is:

```python
TRAIN_SAMPLER = "random"
```

In distributed training, relation-balanced sampling is disabled and the code falls back to `DistributedSampler`.

The Kaggle notebook enables `relation_balanced` at runtime.

---

### 16. Training loop and checkpointing

The standalone training script supports:

- single-process training,
- multi-GPU DDP,
- early stopping,
- gradient accumulation,
- best checkpoint saving,
- optional per-epoch checkpoint saving.

The default checkpoint selection metric is:

```python
CHECKPOINT_METRIC = "mrr_macro_f1"
```

This combines:

```text
0.7 × MRR + 0.3 × Macro-F1
```

using filtered MRR when available.

---

### 17. Evaluation

Evaluation reports raw ranking metrics:

```text
MR
MRR
Hits@1
Hits@3
Hits@5
Hits@10
```

When enabled, it also reports filtered relation-ranking metrics:

```text
FilteredMRR
FilteredHits@1
FilteredHits@3
FilteredHits@5
FilteredHits@10
```

Filtered evaluation masks other known true relations for the same `(head, tail)` pair while keeping the gold relation unmasked.

The code also reports classification metrics:

```text
Accuracy
MacroF1
WeightedF1
```

and efficiency/context statistics:

```text
EvalSeconds
ExamplesPerSecond
AvgInputTokens
AvgHeadContextItems
AvgTailContextItems
AvgPairContextItems
AvgPathItems
```

---

### 18. Relation-frequency bucket metrics

The evaluator groups relations into frequency buckets using training counts:

```text
rare:     count <= RARE_RELATION_THRESHOLD
medium:   count <= MEDIUM_RELATION_THRESHOLD
frequent: count > MEDIUM_RELATION_THRESHOLD
```

Default thresholds:

```python
RARE_RELATION_THRESHOLD = 10
MEDIUM_RELATION_THRESHOLD = 100
```

For each bucket, the code reports:

```text
count
MRR
Hits@1
Hits@3
Hits@10
```

---

### 19. Output files

The code writes:

```text
relation_val_test_results.txt
relation_val_metrics.jsonl
relation_test_results.txt
relation_metrics.jsonl
run_config.json
checkpoint_best.pth
optional checkpoint_epoch_*.pth
saved HuggingFace model/tokenizer files
```

The JSONL files store aggregate metrics. Full per-relation F1 and confusion matrix export are not yet implemented.

---

## Not implemented yet

The following are still future extensions:

1. Relation-prototype classifier fusion.
2. Graph-feature fusion MLP after `[CLS]`.
3. Teacher-student context distillation.
4. Learned context policy.
5. Full per-relation metric export.
6. Full confusion matrix export.
7. Multi-hop paths beyond two hops.
8. A clean ablation runner script.
9. A direct benchmark script for comparing runtime/VRAM against FMS or other baselines.

---

## Suggested ablation order

Run ablations in this order:

| Run | Configuration | Goal |
|---:|---|---|
| A | `[HEAD] + [TAIL]` only | Clean transformer baseline |
| B | + text normalization | Encoder signal |
| C | + adaptive head/tail context | Local graph evidence |
| D | + grouped context | Token compression |
| E | + pair context | Pair-specific evidence |
| F | + schema tokens | Induced structural type information |
| G | + soft schema prior | Logit-level schema guidance |
| H | + path context | Optional explicit path evidence |
| I | + deferred weighted CE | Rare-relation handling |
| J | + relation-balanced sampler | Rare-relation exposure |

For paper reporting, use the same dataset split, hardware, random seed, and evaluation protocol for all ablations.

---

## Validation performed in this check

The current uploaded Python files were checked with:

```text
python -m py_compile config.py data_loader.py model.py train.py utils.py
```

All five files compiled successfully.

A small synthetic data-loader/evaluation check verified that:

1. target-triple leakage is excluded from head/tail context,
2. pair context is generated,
3. dynamic padding collation works,
4. filtered relation masking removes other true relations for the same pair.

Full training was not run in this environment because the real datasets and pretrained model cache are not available here.
