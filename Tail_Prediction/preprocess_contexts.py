import json
import os
import time
from collections import Counter, defaultdict
import math

import pandas as pd
import torch
from transformers import DistilBertTokenizer

import config_tail
from reproducibility import save_json
from sampling import budget_contexts


def load_triplets(file_path):
    return pd.read_csv(file_path, sep="\t", header=None, names=["head", "relation", "tail"])


def build_degrees(triplets):
    degrees = Counter()
    for head, relation, tail in triplets[["head", "relation", "tail"]].itertuples(index=False):
        degrees[head] += 1
        degrees[tail] += 1
    return degrees


def build_context_candidates(train_triplets):
    head_to_neighbors = defaultdict(list)
    relation_to_triples = defaultdict(list)

    for head, relation, tail in train_triplets[["head", "relation", "tail"]].itertuples(index=False):
        head_to_neighbors[head].append((head, relation, tail))
        relation_to_triples[relation].append((head, relation, tail))

    return head_to_neighbors, relation_to_triples


def exclude_exact(triples, head, relation, tail):
    return [triple for triple in triples if triple != (head, relation, tail)]


def compute_avg(values, count):
    return sum(values) / count if count else 0.0


def build_split_contexts(
    split_name,
    split_triplets,
    head_to_neighbors,
    relation_to_triples,
    degrees,
    tail_to_idx,
    tokenizer,
    output_path,
    tokenized_cache_path=None,
):
    hc_lengths = []
    rc_lengths = []
    token_lengths = []
    at_max_count = 0
    tokenized_records = [] if tokenized_cache_path else None
    total_rows = len(split_triplets)
    progress_every = max(1, total_rows // 10) if total_rows else 1

    print(
        f"[preprocess] Building {split_name} contexts: {total_rows} triples -> {output_path}",
        flush=True,
    )

    processed_dir = os.path.dirname(output_path)
    progress_path = os.path.join(processed_dir, f"{split_name}_progress.json")

    with open(output_path, "w", encoding="utf-8") as outfile:
        for idx, (head, relation, tail) in enumerate(
            split_triplets[["head", "relation", "tail"]].itertuples(index=False), start=1
        ):
            hc_candidates = exclude_exact(head_to_neighbors.get(head, []), head, relation, tail)
            rc_candidates = exclude_exact(relation_to_triples.get(relation, []), head, relation, tail)

            hc_selected, rc_selected = budget_contexts(
                hc_candidates,
                rc_candidates,
                degrees,
                relation,
                config_tail.MAX_HC,
                config_tail.MAX_RC,
                config_tail.MAX_TOTAL_CONTEXT,
                config_tail.MAX_RC_IF_HC_SHORT,
                config_tail.MAX_SAME_RELATION_IN_HC,
                entity_types=None,
            )

            head_context = [f"{h}-{r}-{t}" for h, r, t in hc_selected]
            relation_context = [f"{h}-{r}-{t}" for h, r, t in rc_selected]

            head_context_str = " ".join(head_context)
            relation_context_str = " ".join(relation_context)

            if config_tail.CONTEXT_ORDER == "prioritize_relation":
                matched_hc = [f"{h}-{r}-{t}" for h, r, t in hc_selected if r == relation]
                other_hc = [f"{h}-{r}-{t}" for h, r, t in hc_selected if r != relation]
                matched_hc_str = " ".join(matched_hc)
                other_hc_str = " ".join(other_hc)
                input_text = (
                    f"{head} [SEP] {relation} [SEP] {matched_hc_str} [SEP] "
                    f"{other_hc_str} [SEP] {relation_context_str}"
                ).strip()
            else:
                input_text = (
                    f"{head} [SEP] {head_context_str} [SEP] "
                    f"{relation} [SEP] {relation_context_str}"
                ).strip()

            encoded = tokenizer(
                input_text,
                truncation=True,
                max_length=config_tail.MAX_LENGTH,
                add_special_tokens=True,
                return_attention_mask=True,
            )
            encoded_len = len(encoded["input_ids"])
            token_lengths.append(encoded_len)
            if encoded_len >= config_tail.MAX_LENGTH:
                at_max_count += 1

            if tokenized_records is not None:
                tokenized_records.append(
                    {
                        "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                        "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                        "label": tail_to_idx[tail],
                        "metadata": {"head": head, "relation": relation, "tail": tail},
                    }
                )

            record = {
                "head": head,
                "relation": relation,
                "tail": tail,
                "head_context": head_context,
                "relation_context": relation_context,
                "input_text": input_text,
                "label": tail_to_idx[tail],
            }

            outfile.write(json.dumps(record) + "\n")

            hc_lengths.append(len(head_context))
            rc_lengths.append(len(relation_context))

            if idx % progress_every == 0 or idx == total_rows:
                pct = (idx / total_rows) * 100.0 if total_rows else 100.0
                print(
                    f"[preprocess] {split_name}: {idx}/{total_rows} ({pct:.1f}%)",
                    flush=True,
                )
                # Persist incremental tokenized cache and progress so work is not lost
                if tokenized_cache_path and tokenized_records:
                    try:
                        torch.save(tokenized_records, tokenized_cache_path)
                        with open(progress_path, "w", encoding="utf-8") as pf:
                            pf.write(json.dumps({"processed_rows": idx}))
                    except Exception:
                        # Best-effort; do not fail preprocessing on save errors
                        pass

    stats = {
        f"avg_hc_len_{split_name}": compute_avg(hc_lengths, len(hc_lengths)),
        f"avg_rc_len_{split_name}": compute_avg(rc_lengths, len(rc_lengths)),
        f"avg_input_tokens_{split_name}": compute_avg(token_lengths, len(token_lengths)),
        f"pct_input_tokens_ge_max_{split_name}": (at_max_count / len(token_lengths)) * 100.0
        if token_lengths
        else 0.0,
    }

    if tokenized_cache_path:
        try:
            torch.save(tokenized_records, tokenized_cache_path)
            print(
                f"[preprocess] Saved tokenized cache: {tokenized_cache_path}",
                flush=True,
            )
            # write final progress
            try:
                with open(os.path.join(processed_dir, f"{split_name}_progress.json"), "w", encoding="utf-8") as pf:
                    pf.write(json.dumps({"processed_rows": total_rows}))
            except Exception:
                pass
        except Exception:
            print(f"[preprocess] Warning: failed to save tokenized cache: {tokenized_cache_path}", flush=True)

    return stats


def build_tokenized_cache_from_jsonl(jsonl_path, tokenizer, output_path):
    tokenized_records = []
    total_rows = 0
    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            total_rows += 1
            record = json.loads(line)
            encoded = tokenizer(
                record["input_text"],
                truncation=True,
                max_length=config_tail.MAX_LENGTH,
                add_special_tokens=True,
                return_attention_mask=True,
            )
            tokenized_records.append(
                {
                    "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                    "label": record["label"],
                    "metadata": {
                        "head": record["head"],
                        "relation": record["relation"],
                        "tail": record["tail"],
                    },
                }
            )

    torch.save(tokenized_records, output_path)
    print(
        f"[preprocess] Built tokenized cache from JSONL: {output_path} (rows={total_rows})",
        flush=True,
    )


def preprocess_all():
    os.makedirs(config_tail.PROCESSED_DIR, exist_ok=True)

    print("[preprocess] Loading input triples...", flush=True)

    train_triplets = load_triplets(config_tail.train_file_path)
    valid_triplets = load_triplets(config_tail.valid_file_path)
    test_triplets = load_triplets(config_tail.test_file_path)

    print(
        f"[preprocess] Loaded train={len(train_triplets)}, valid={len(valid_triplets)}, test={len(test_triplets)}",
        flush=True,
    )

    all_triplets = pd.concat([train_triplets, valid_triplets, test_triplets], ignore_index=True)

    all_entities = sorted(set(all_triplets["head"]).union(set(all_triplets["tail"])))
    all_relations = sorted(all_triplets["relation"].unique().tolist())
    tail_labels = sorted(all_triplets["tail"].unique().tolist())

    entity_to_idx = {entity: idx for idx, entity in enumerate(all_entities)}
    relation_to_idx = {rel: idx for idx, rel in enumerate(all_relations)}
    tail_to_idx = {tail: idx for idx, tail in enumerate(tail_labels)}

    save_json(os.path.join(config_tail.PROCESSED_DIR, "entity_vocab.json"), all_entities)
    save_json(os.path.join(config_tail.PROCESSED_DIR, "relation_vocab.json"), all_relations)
    save_json(os.path.join(config_tail.PROCESSED_DIR, "tail_label_vocab.json"), tail_labels)

    degrees = build_degrees(train_triplets)
    head_to_neighbors, relation_to_triples = build_context_candidates(train_triplets)

    head_neighbor_counts = [len(neighbors) for neighbors in head_to_neighbors.values()]
    unique_train_heads = train_triplets["head"].nunique()

    tokenizer = DistilBertTokenizer.from_pretrained(config_tail.MODEL_NAME)
    print(f"[preprocess] Tokenizer ready: {config_tail.MODEL_NAME}", flush=True)

    stats = {
        "num_train_triples": len(train_triplets),
        "num_valid_triples": len(valid_triplets),
        "num_test_triples": len(test_triplets),
        "num_entities": len(all_entities),
        "num_relations": len(all_relations),
        "num_tail_labels": len(tail_labels),
        "unique_tails_train": train_triplets["tail"].nunique(),
        "unique_tails_valid": valid_triplets["tail"].nunique(),
        "unique_tails_test": test_triplets["tail"].nunique(),
        "avg_head_neighbors_train": compute_avg(head_neighbor_counts, len(head_neighbor_counts)),
        "pct_heads_with_context_train": (len(head_neighbor_counts) / unique_train_heads) * 100.0
        if unique_train_heads
        else 0.0,
    }

    train_tails = set(train_triplets["tail"].unique().tolist())
    valid_tails = set(valid_triplets["tail"].unique().tolist())
    test_tails = set(test_triplets["tail"].unique().tolist())
    unseen_valid_tails = valid_tails - train_tails
    unseen_test_tails = test_tails - train_tails
    stats["num_valid_tails_unseen"] = len(unseen_valid_tails)
    stats["num_test_tails_unseen"] = len(unseen_test_tails)
    stats["pct_valid_tails_unseen"] = (
        (len(unseen_valid_tails) / len(valid_tails)) * 100.0 if valid_tails else 0.0
    )
    stats["pct_test_tails_unseen"] = (
        (len(unseen_test_tails) / len(test_tails)) * 100.0 if test_tails else 0.0
    )

    context_config = {
        "max_hc": config_tail.MAX_HC,
        "max_rc": config_tail.MAX_RC,
        "max_total_context": config_tail.MAX_TOTAL_CONTEXT,
        "max_rc_if_hc_short": config_tail.MAX_RC_IF_HC_SHORT,
        "max_same_relation_in_hc": config_tail.MAX_SAME_RELATION_IN_HC,
        "max_length": config_tail.MAX_LENGTH,
        "context_order": config_tail.CONTEXT_ORDER,
        "context_graph": "train_only",
        "exclude_current_training_triple_from_context": True,
    }
    save_json(os.path.join(config_tail.PROCESSED_DIR, "context_config.json"), context_config)

    prior_path = os.path.join(config_tail.PROCESSED_DIR, "relation_tail_prior.json")
    relation_counts = defaultdict(Counter)
    relation_totals = Counter()
    for head, relation, tail in train_triplets[["head", "relation", "tail"]].itertuples(index=False):
        relation_counts[relation][tail] += 1
        relation_totals[relation] += 1

    prior = {}
    for relation, tail_counts in relation_counts.items():
        total = relation_totals[relation]
        if total == 0:
            continue
        prior[relation] = {tail: math.log(count / total) for tail, count in tail_counts.items()}

    save_json(prior_path, prior)

    train_path = os.path.join(config_tail.PROCESSED_DIR, "train_context.jsonl")
    valid_path = os.path.join(config_tail.PROCESSED_DIR, "valid_context.jsonl")
    test_path = os.path.join(config_tail.PROCESSED_DIR, "test_context.jsonl")

    tokenized_train_path = os.path.join(config_tail.PROCESSED_DIR, "train_tokenized.pt")
    tokenized_valid_path = os.path.join(config_tail.PROCESSED_DIR, "valid_tokenized.pt")
    tokenized_test_path = os.path.join(config_tail.PROCESSED_DIR, "test_tokenized.pt")
    tokenized_paths = (tokenized_train_path, tokenized_valid_path, tokenized_test_path)
    tokenized_enabled = config_tail.SAVE_TOKENIZED_CACHE

    stats.update(
        build_split_contexts(
            "train",
            train_triplets,
            head_to_neighbors,
            relation_to_triples,
            degrees,
            tail_to_idx,
            tokenizer,
            train_path,
            tokenized_cache_path=tokenized_train_path if tokenized_enabled else None,
        )
    )
    stats.update(
        build_split_contexts(
            "valid",
            valid_triplets,
            head_to_neighbors,
            relation_to_triples,
            degrees,
            tail_to_idx,
            tokenizer,
            valid_path,
            tokenized_cache_path=tokenized_valid_path if tokenized_enabled else None,
        )
    )
    stats.update(
        build_split_contexts(
            "test",
            test_triplets,
            head_to_neighbors,
            relation_to_triples,
            degrees,
            tail_to_idx,
            tokenizer,
            test_path,
            tokenized_cache_path=tokenized_test_path if tokenized_enabled else None,
        )
    )

    save_json(os.path.join(config_tail.PROCESSED_DIR, "dataset_stats.json"), stats)
    if tokenized_enabled:
        stats["tokenized_cache_paths"] = list(tokenized_paths)

    print("[preprocess] Finished. Saved vocabularies, contexts, and dataset_stats.json", flush=True)

    if stats.get("avg_hc_len_train", 0.0) < 0.1:
        print(
            (
                "[preprocess] Warning: avg_hc_len_train is near zero. "
                f"avg_head_neighbors_train={stats.get('avg_head_neighbors_train', 0.0):.2f}, "
                f"pct_heads_with_context_train={stats.get('pct_heads_with_context_train', 0.0):.2f}%"
            ),
            flush=True,
        )

    return {
        "train_path": train_path,
        "valid_path": valid_path,
        "test_path": test_path,
        "entity_to_idx": entity_to_idx,
        "relation_to_idx": relation_to_idx,
        "tail_to_idx": tail_to_idx,
        "dataset_stats": stats,
    }


if __name__ == "__main__":
    start = time.perf_counter()
    preprocess_all()
    elapsed = time.perf_counter() - start
    print(f"Preprocessing finished in {elapsed:.2f}s")
