import json
import os
import time
from collections import Counter, defaultdict

import pandas as pd
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
):
    hc_lengths = []
    rc_lengths = []
    token_lengths = []

    with open(output_path, "w", encoding="utf-8") as outfile:
        for head, relation, tail in split_triplets[["head", "relation", "tail"]].itertuples(index=False):
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

            input_text = (
                f"{head} [SEP] {head_context_str} [SEP] "
                f"{relation} [SEP] {relation_context_str}"
            ).strip()

            encoded = tokenizer(
                input_text,
                truncation=True,
                max_length=config_tail.MAX_LENGTH,
                add_special_tokens=True,
            )
            token_lengths.append(len(encoded["input_ids"]))

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

    stats = {
        f"avg_hc_len_{split_name}": compute_avg(hc_lengths, len(hc_lengths)),
        f"avg_rc_len_{split_name}": compute_avg(rc_lengths, len(rc_lengths)),
        f"avg_input_tokens_{split_name}": compute_avg(token_lengths, len(token_lengths)),
    }

    return stats


def preprocess_all():
    os.makedirs(config_tail.PROCESSED_DIR, exist_ok=True)

    train_triplets = load_triplets(config_tail.train_file_path)
    valid_triplets = load_triplets(config_tail.valid_file_path)
    test_triplets = load_triplets(config_tail.test_file_path)

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

    tokenizer = DistilBertTokenizer.from_pretrained(config_tail.MODEL_NAME)

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
    }

    train_tails = set(train_triplets["tail"].unique().tolist())
    test_tails = set(test_triplets["tail"].unique().tolist())
    unseen_test_tails = test_tails - train_tails
    stats["num_test_tails_unseen"] = len(unseen_test_tails)
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
        "context_graph": "train_only",
        "exclude_current_training_triple_from_context": True,
    }
    save_json(os.path.join(config_tail.PROCESSED_DIR, "context_config.json"), context_config)

    train_path = os.path.join(config_tail.PROCESSED_DIR, "train_context.jsonl")
    valid_path = os.path.join(config_tail.PROCESSED_DIR, "valid_context.jsonl")
    test_path = os.path.join(config_tail.PROCESSED_DIR, "test_context.jsonl")

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
        )
    )

    save_json(os.path.join(config_tail.PROCESSED_DIR, "dataset_stats.json"), stats)

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
