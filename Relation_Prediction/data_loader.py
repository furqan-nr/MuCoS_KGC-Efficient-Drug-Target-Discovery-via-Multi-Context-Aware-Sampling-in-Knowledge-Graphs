"""Data loading and context construction for general KG relation prediction.

This module implements a negative-sample-free, general-KG relation-prediction
pipeline with:
- train-only graph statistics,
- target-triple exclusion to prevent relation-label leakage,
- generic entity/relation text normalization and optional label maps,
- role-aware bidirectional context,
- grouped/compressed context by relation,
- adaptive/relevance-aware context selection,
- pair-specific context for (head, ?, tail),
- compressed 2-hop relation motifs,
- induced schema signatures and train-only schema relation priors,
- dynamic-padding-friendly dataset outputs.
"""

from __future__ import annotations

import math
import os
import re
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


Triple = Tuple[str, str, str]
Neighbor = Tuple[str, str]  # relation, neighbor entity
EdgeItem = Tuple[str, str, str]  # direction, relation, neighbor entity
PathItem = Tuple[str, str, str, str, str]  # r1, dir1, middle, r2, dir2


# ==================== BASIC LOADING ====================

def load_triplets(file_path: str) -> pd.DataFrame:
    """Load triples from a tab-separated file with columns head, relation, tail."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Triplet file not found: {file_path}")
    df = pd.read_csv(file_path, sep="\t", header=None, names=["head", "relation", "tail"], dtype=str)
    df = df.dropna(subset=["head", "relation", "tail"]).reset_index(drop=True)
    return df


def load_entity_types(entity_type_file: Optional[str], default_type: str = "UNK") -> Dict[str, str]:
    """Load optional TSV mapping: entity_id<TAB>entity_type."""
    if not entity_type_file:
        return {}
    if not os.path.exists(entity_type_file):
        raise FileNotFoundError(f"ENTITY_TYPE_FILE not found: {entity_type_file}")
    type_df = pd.read_csv(entity_type_file, sep="\t", header=None, names=["entity", "type"], dtype=str)
    type_df = type_df.dropna(subset=["entity", "type"])
    return dict(zip(type_df["entity"], type_df["type"]))


def load_label_map(label_file: Optional[str]) -> Dict[str, str]:
    """Load optional TSV mapping: raw_id<TAB>readable label.

    The mapping is used only for encoder text. Raw IDs remain in metadata and
    graph structures.
    """
    if not label_file:
        return {}
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label map file not found: {label_file}")
    df = pd.read_csv(label_file, sep="\t", header=None, names=["id", "label"], dtype=str)
    df = df.dropna(subset=["id", "label"])
    return dict(zip(df["id"], df["label"]))


def get_entity_type(entity: str, entity_types: Mapping[str, str], default_type: str = "UNK") -> str:
    return entity_types.get(entity, default_type)


# ==================== TEXT NORMALIZATION ====================

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _strip_uri_like(value: str) -> str:
    value = str(value)
    value = value.strip()
    # Common wrappers in YAGO/Wikidata/Freebase dumps.
    value = value.strip("<>")
    value = value.replace("http://", " ").replace("https://", " ")
    value = re.sub(r"www\.", " ", value)
    # Keep the meaningful tail of slash/hash-separated IDs but preserve relation
    # paths as text if they contain multiple components.
    value = value.replace("/", " ").replace("#", " ").replace(":", " ")
    return value


def normalize_kg_text(raw: str, dataset_name: str = "", label_map: Optional[Mapping[str, str]] = None) -> str:
    """Convert KG IDs/URIs/synsets to readable text for the language model."""
    if raw is None:
        return ""
    raw = str(raw)
    if label_map and raw in label_map:
        raw = str(label_map[raw])

    text = raw
    ds = (dataset_name or "").lower()

    # WordNet examples often look like __good_a_01 or good.a.01.
    if "wn" in ds or "wordnet" in ds:
        text = text.replace("__", " ")
        text = text.replace(".", " ")

    text = _strip_uri_like(text)
    text = text.replace("_", " ").replace("-", " ")
    text = _CAMEL_RE.sub(" ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if text else str(raw).lower()


# ==================== RELATION LABEL SPACE ====================

def build_relation_to_idx(train_triplets: pd.DataFrame) -> Dict[str, int]:
    """Build classifier label space from training relations only."""
    relations = sorted(train_triplets["relation"].unique().tolist())
    return {rel: idx for idx, rel in enumerate(relations)}


def validate_relation_coverage(
    relation_to_idx: Mapping[str, int],
    valid_triplets: pd.DataFrame,
    test_triplets: pd.DataFrame,
) -> None:
    """Fail early if validation/test contains unseen relation labels."""
    train_relations = set(relation_to_idx.keys())
    valid_unseen = sorted(set(valid_triplets["relation"].unique()) - train_relations)
    test_unseen = sorted(set(test_triplets["relation"].unique()) - train_relations)
    if valid_unseen or test_unseen:
        raise ValueError(
            "Relation labels in validation/test must exist in train for closed-world "
            f"relation classification. valid_unseen={valid_unseen}, test_unseen={test_unseen}"
        )


# ==================== GRAPH PRECOMPUTATION ====================

def _to_plain_dict_of_lists(mapping: Mapping[str, Sequence]) -> Dict[str, List]:
    return {key: list(value) for key, value in mapping.items()}


def _degree_bucket(degree: int) -> str:
    if degree <= 1:
        return "d0_1"
    if degree <= 3:
        return "d2_3"
    if degree <= 10:
        return "d4_10"
    if degree <= 30:
        return "d11_30"
    return "d31_plus"


def _entity_schema_signature(
    entity: str,
    outgoing_map: Mapping[str, Sequence[Neighbor]],
    incoming_map: Mapping[str, Sequence[Neighbor]],
    entity_degrees: Mapping[str, int],
    top_k: int = 3,
    mode: str = "relation_signature",
) -> str:
    degree = int(entity_degrees.get(entity, 0))
    if mode == "degree":
        return _degree_bucket(degree)

    out_counts = Counter(rel for rel, _ in outgoing_map.get(entity, []))
    in_counts = Counter(rel for rel, _ in incoming_map.get(entity, []))
    top_out = ",".join(rel for rel, _c in out_counts.most_common(top_k)) or "none"
    top_in = ",".join(rel for rel, _c in in_counts.most_common(top_k)) or "none"
    return f"{_degree_bucket(degree)}|O:{top_out}|I:{top_in}"


def precompute_graph_info(
    train_triplets: pd.DataFrame,
    entity_types: Optional[Mapping[str, str]] = None,
    default_type: str = "UNK",
    schema_top_relations: int = 3,
    schema_mode: str = "relation_signature",
) -> Dict[str, object]:
    """Precompute train-only graph structures used by context sampling.

    No validation/test triples are used here. This prevents validation/test graph
    leakage while allowing filtered evaluation to be handled separately.
    """
    entity_types = entity_types or {}

    entity_degrees: Counter = Counter()
    out_degrees: Counter = Counter()
    in_degrees: Counter = Counter()
    relation_counts: Counter = Counter()
    outgoing_map: Dict[str, List[Neighbor]] = defaultdict(list)
    incoming_map: Dict[str, List[Neighbor]] = defaultdict(list)
    undirected_map: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    type_pair_to_relations: Dict[Tuple[str, str], set] = defaultdict(set)
    pair_to_relations: Dict[Tuple[str, str], set] = defaultdict(set)
    triples_set = set()
    train_triples: List[Triple] = []

    for row in train_triplets.itertuples(index=False):
        head, relation, tail = str(row.head), str(row.relation), str(row.tail)
        triple = (head, relation, tail)
        train_triples.append(triple)
        triples_set.add(triple)
        entity_degrees[head] += 1
        entity_degrees[tail] += 1
        out_degrees[head] += 1
        in_degrees[tail] += 1
        relation_counts[relation] += 1

        outgoing_map[head].append((relation, tail))
        incoming_map[tail].append((relation, head))
        undirected_map[head].append((tail, relation, "out"))
        undirected_map[tail].append((head, relation, "in"))
        pair_to_relations[(head, tail)].add(relation)

        h_type = get_entity_type(head, entity_types, default_type)
        t_type = get_entity_type(tail, entity_types, default_type)
        type_pair_to_relations[(h_type, t_type)].add(relation)

    all_entities = sorted(set(entity_degrees.keys()))
    entity_schema = {
        entity: _entity_schema_signature(
            entity,
            outgoing_map,
            incoming_map,
            entity_degrees,
            top_k=schema_top_relations,
            mode=schema_mode,
        )
        for entity in all_entities
    }

    schema_pair_relation_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for head, relation, tail in train_triples:
        schema_pair = (entity_schema.get(head, "UNK_SCHEMA"), entity_schema.get(tail, "UNK_SCHEMA"))
        schema_pair_relation_counts[schema_pair][relation] += 1

    relation_counts_dict = dict(relation_counts)
    total_relations = max(1, sum(relation_counts_dict.values()))

    return {
        "entity_degrees": dict(entity_degrees),
        "out_degrees": dict(out_degrees),
        "in_degrees": dict(in_degrees),
        "relation_counts": relation_counts_dict,
        "total_relation_mentions": total_relations,
        "outgoing_map": _to_plain_dict_of_lists(outgoing_map),
        "incoming_map": _to_plain_dict_of_lists(incoming_map),
        "undirected_map": _to_plain_dict_of_lists(undirected_map),
        "type_pair_to_relations": {k: sorted(v) for k, v in type_pair_to_relations.items()},
        "pair_to_relations": {k: sorted(v) for k, v in pair_to_relations.items()},
        "schema_pair_relation_counts": {k: dict(v) for k, v in schema_pair_relation_counts.items()},
        "entity_schema": entity_schema,
        "triples_set": triples_set,
        "entities": all_entities,
    }


# Backward-compatible alias for old code.
def precompute_entity_info(all_triplets: pd.DataFrame, max_degree: int):
    graph_info = precompute_graph_info(all_triplets)
    entity_contexts = {}
    for entity in graph_info["entities"]:
        out_items = [f"[OUT] {rel} {nbr}" for rel, nbr in graph_info["outgoing_map"].get(entity, [])[:max_degree]]
        in_items = [f"[IN] {rel} {nbr}" for rel, nbr in graph_info["incoming_map"].get(entity, [])[:max_degree]]
        entity_contexts[entity] = " ".join(out_items + in_items)
    return graph_info["entity_degrees"], entity_contexts


def build_known_true_relations_by_pair(
    train_triplets: pd.DataFrame,
    valid_triplets: pd.DataFrame,
    test_triplets: pd.DataFrame,
) -> Dict[Tuple[str, str], List[str]]:
    """Build all known true relations for each (head, tail) pair for filtered eval.

    This is used only for filtered ranking metrics, not for training context.
    """
    pair_to_relations: Dict[Tuple[str, str], set] = defaultdict(set)
    for df in (train_triplets, valid_triplets, test_triplets):
        for row in df.itertuples(index=False):
            pair_to_relations[(str(row.head), str(row.tail))].add(str(row.relation))
    return {pair: sorted(rels) for pair, rels in pair_to_relations.items()}


# ==================== CONTEXT SAMPLING ====================

def adaptive_budget(
    degree: int,
    mode: str = "adaptive",
    min_context: int = 5,
    base_context: int = 15,
    max_context: int = 30,
) -> int:
    """Choose context budget. Sparse nodes get more context, hubs get less."""
    if mode == "fixed":
        return max(1, base_context)
    if degree <= 2:
        return max_context
    if degree <= 10:
        return base_context
    reduced = int(base_context - math.log1p(degree))
    return max(min_context, min(max_context, reduced))


def _relation_rarity(relation: str, relation_counts: Mapping[str, int], total_mentions: int) -> float:
    return math.log1p(total_mentions / (1.0 + relation_counts.get(relation, 0)))


def rank_neighbors(
    neighbors: Sequence[Neighbor],
    graph_info: Mapping[str, object],
    w_degree: float = 1.0,
    w_rel_rarity: float = 0.35,
    w_redundancy: float = 0.15,
) -> List[Neighbor]:
    """Rank neighbors by density + relation rarity - redundancy."""
    entity_degrees: Mapping[str, int] = graph_info["entity_degrees"]
    relation_counts: Mapping[str, int] = graph_info["relation_counts"]
    total_mentions: int = graph_info["total_relation_mentions"]
    rel_seen = Counter(rel for rel, _ in neighbors)

    def score(item: Neighbor) -> Tuple[float, str, str]:
        rel, neighbor = item
        degree_score = math.log1p(entity_degrees.get(neighbor, 0))
        rarity_score = _relation_rarity(rel, relation_counts, total_mentions)
        redundancy_penalty = math.log1p(rel_seen[rel])
        value = (w_degree * degree_score) + (w_rel_rarity * rarity_score) - (w_redundancy * redundancy_penalty)
        return (-value, neighbor, rel)

    return sorted(neighbors, key=score)


def pack_context_items(tokenizer, items: Sequence[str], max_tokens: int) -> List[str]:
    """Pack context strings under an approximate tokenizer-token budget."""
    if max_tokens <= 0:
        return []
    selected: List[str] = []
    used = 0
    for item in items:
        token_count = len(tokenizer.tokenize(item))
        if token_count == 0:
            continue
        if used + token_count > max_tokens:
            break
        selected.append(item)
        used += token_count
    return selected


def _edge_matches_excluded_triple(anchor: str, relation: str, neighbor: str, direction: str, exclude_triple: Optional[Triple]) -> bool:
    if exclude_triple is None:
        return False
    h, r, t = exclude_triple
    if direction == "out":
        return anchor == h and relation == r and neighbor == t
    if direction == "in":
        return neighbor == h and relation == r and anchor == t
    return False


def _group_edges_by_relation(
    entity_role: str,
    edges: Sequence[EdgeItem],
    graph_info: Mapping[str, object],
    dataset_name: str,
    entity_label_map: Mapping[str, str],
    relation_label_map: Mapping[str, str],
    normalize_entity_text: bool,
    normalize_relation_text: bool,
    max_neighbors_per_relation: int = 3,
) -> List[str]:
    """Compress repeated relation context items into one item per relation/direction."""
    out_token = "[H_OUT]" if entity_role == "head" else "[T_OUT]"
    in_token = "[H_IN]" if entity_role == "head" else "[T_IN]"
    grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for direction, rel, nbr in edges:
        token = out_token if direction == "out" else in_token
        grouped[(token, rel)].append(nbr)

    entity_degrees: Mapping[str, int] = graph_info["entity_degrees"]
    items: List[str] = []
    for (token, rel), nbrs in grouped.items():
        nbrs_sorted = sorted(nbrs, key=lambda n: (-entity_degrees.get(n, 0), n))[:max_neighbors_per_relation]
        rel_text = normalize_kg_text(rel, dataset_name, relation_label_map) if normalize_relation_text else rel
        nbr_texts = [normalize_kg_text(n, dataset_name, entity_label_map) if normalize_entity_text else n for n in nbrs_sorted]
        items.append(f"{token} {rel_text}: {' | '.join(nbr_texts)}")
    return items


def build_entity_context_items(
    entity: str,
    role: str,
    graph_info: Mapping[str, object],
    entity_types: Mapping[str, str],
    dataset_name: str = "",
    entity_label_map: Optional[Mapping[str, str]] = None,
    relation_label_map: Optional[Mapping[str, str]] = None,
    normalize_entity_text: bool = True,
    normalize_relation_text: bool = True,
    default_type: str = "UNK",
    context_mode: str = "adaptive",
    min_context: int = 5,
    base_context: int = 15,
    max_context: int = 30,
    use_second_hop: bool = False,
    second_hop_budget: int = 5,
    w_degree: float = 1.0,
    w_rel_rarity: float = 0.35,
    w_redundancy: float = 0.15,
    exclude_triple: Optional[Triple] = None,
    group_context_by_relation: bool = True,
    max_neighbors_per_relation: int = 3,
) -> List[str]:
    """Build role-aware context items for one entity."""
    entity_label_map = entity_label_map or {}
    relation_label_map = relation_label_map or {}
    degree = int(graph_info["entity_degrees"].get(entity, 0))
    budget = adaptive_budget(degree, context_mode, min_context, base_context, max_context)
    out_token = "[H_OUT]" if role == "head" else "[T_OUT]"
    in_token = "[H_IN]" if role == "head" else "[T_IN]"

    outgoing: List[EdgeItem] = [
        ("out", rel, nbr)
        for rel, nbr in graph_info["outgoing_map"].get(entity, [])
        if not _edge_matches_excluded_triple(entity, rel, nbr, "out", exclude_triple)
    ]
    incoming: List[EdgeItem] = [
        ("in", rel, nbr)
        for rel, nbr in graph_info["incoming_map"].get(entity, [])
        if not _edge_matches_excluded_triple(entity, rel, nbr, "in", exclude_triple)
    ]

    combined = outgoing + incoming
    ranked_pairs = rank_neighbors([(rel, nbr) for _, rel, nbr in combined], graph_info, w_degree, w_rel_rarity, w_redundancy)
    ranked_lookup = {(rel, nbr): idx for idx, (rel, nbr) in enumerate(ranked_pairs)}
    combined_sorted = sorted(combined, key=lambda item: ranked_lookup.get((item[1], item[2]), 10**12))[:budget]

    if group_context_by_relation:
        items = _group_edges_by_relation(
            role,
            combined_sorted,
            graph_info,
            dataset_name,
            entity_label_map,
            relation_label_map,
            normalize_entity_text,
            normalize_relation_text,
            max_neighbors_per_relation=max_neighbors_per_relation,
        )
    else:
        items = []
        for direction, rel, nbr in combined_sorted:
            token = out_token if direction == "out" else in_token
            rel_text = normalize_kg_text(rel, dataset_name, relation_label_map) if normalize_relation_text else rel
            nbr_text = normalize_kg_text(nbr, dataset_name, entity_label_map) if normalize_entity_text else nbr
            nbr_type = get_entity_type(nbr, entity_types, default_type)
            items.append(f"{token} {rel_text} {nbr_text} [TYPE] {nbr_type}")

    if use_second_hop and second_hop_budget > 0:
        expanded = 0
        for _direction, _rel, nbr in combined_sorted:
            if expanded >= second_hop_budget:
                break
            second_neighbors = (
                [("out", rel2, nbr2) for rel2, nbr2 in graph_info["outgoing_map"].get(nbr, [])[:1]]
                + [("in", rel2, nbr2) for rel2, nbr2 in graph_info["incoming_map"].get(nbr, [])[:1]]
            )
            for dir2, rel2, nbr2 in second_neighbors:
                if expanded >= second_hop_budget:
                    break
                if _edge_matches_excluded_triple(nbr, rel2, nbr2, dir2, exclude_triple):
                    continue
                rel_text = normalize_kg_text(rel2, dataset_name, relation_label_map) if normalize_relation_text else rel2
                nbr2_text = normalize_kg_text(nbr2, dataset_name, entity_label_map) if normalize_entity_text else nbr2
                items.append(f"[2HOP] {rel_text} {nbr2_text}")
                expanded += 1

    return items


# ==================== PAIR CONTEXT AND PATHS ====================

def find_two_hop_paths(
    head: str,
    tail: str,
    graph_info: Mapping[str, object],
    max_paths: int = 3,
) -> List[PathItem]:
    """Find cheap undirected 2-hop paths head--middle--tail from train graph."""
    if max_paths <= 0 or head == tail:
        return []

    undirected: Mapping[str, Sequence[Tuple[str, str, str]]] = graph_info["undirected_map"]
    first_hop = undirected.get(head, [])
    tail_neighbors = defaultdict(list)
    for nbr, rel, direction in undirected.get(tail, []):
        tail_neighbors[nbr].append((rel, direction))

    paths: List[PathItem] = []
    for middle, r1, d1 in first_hop:
        if middle == head or middle == tail:
            continue
        if middle not in tail_neighbors:
            continue
        for r2, d2_from_tail in tail_neighbors[middle]:
            d2 = "out" if d2_from_tail == "in" else "in"
            paths.append((r1, d1, middle, r2, d2))

    if not paths:
        return []

    relation_counts: Mapping[str, int] = graph_info["relation_counts"]
    total_mentions: int = graph_info["total_relation_mentions"]

    def score(path: PathItem) -> Tuple[float, str, str, str]:
        r1, _d1, middle, r2, _d2 = path
        rarity = _relation_rarity(r1, relation_counts, total_mentions) + _relation_rarity(r2, relation_counts, total_mentions)
        mid_degree = math.log1p(graph_info["entity_degrees"].get(middle, 0))
        return (-(rarity - 0.05 * mid_degree), middle, r1, r2)

    return sorted(paths, key=score)[:max_paths]


def path_items_to_text(
    head: str,
    tail: str,
    paths: Sequence[PathItem],
    dataset_name: str = "",
    entity_label_map: Optional[Mapping[str, str]] = None,
    relation_label_map: Optional[Mapping[str, str]] = None,
    normalize_entity_text: bool = True,
    normalize_relation_text: bool = True,
    mode: str = "motif",
) -> List[str]:
    entity_label_map = entity_label_map or {}
    relation_label_map = relation_label_map or {}
    h_text = normalize_kg_text(head, dataset_name, entity_label_map) if normalize_entity_text else head
    t_text = normalize_kg_text(tail, dataset_name, entity_label_map) if normalize_entity_text else tail
    text_items: List[str] = []
    for r1, d1, middle, r2, d2 in paths:
        r1_text = normalize_kg_text(r1, dataset_name, relation_label_map) if normalize_relation_text else r1
        r2_text = normalize_kg_text(r2, dataset_name, relation_label_map) if normalize_relation_text else r2
        if mode == "full":
            m_text = normalize_kg_text(middle, dataset_name, entity_label_map) if normalize_entity_text else middle
            text_items.append(f"[PATH] {h_text} -{d1}:{r1_text}-> {m_text} -{d2}:{r2_text}-> {t_text}")
        else:
            text_items.append(f"[PATH] {d1}:{r1_text} -> {d2}:{r2_text}")
    return text_items


def build_pair_context_items(
    head: str,
    tail: str,
    graph_info: Mapping[str, object],
    dataset_name: str = "",
    entity_label_map: Optional[Mapping[str, str]] = None,
    relation_label_map: Optional[Mapping[str, str]] = None,
    normalize_entity_text: bool = True,
    normalize_relation_text: bool = True,
    max_common_neighbors: int = 5,
    max_pair_motifs: int = 5,
    exclude_relation: Optional[str] = None,
) -> List[str]:
    """Build compact pair-specific context for relation prediction."""
    entity_label_map = entity_label_map or {}
    relation_label_map = relation_label_map or {}
    items: List[str] = []
    undirected = graph_info["undirected_map"]
    entity_degrees = graph_info["entity_degrees"]

    h_neighbors = {nbr for nbr, _rel, _d in undirected.get(head, [])}
    t_neighbors = {nbr for nbr, _rel, _d in undirected.get(tail, [])}
    common = sorted(h_neighbors & t_neighbors, key=lambda n: (-entity_degrees.get(n, 0), n))[:max_common_neighbors]
    if common:
        common_text = [normalize_kg_text(n, dataset_name, entity_label_map) if normalize_entity_text else n for n in common]
        items.append(f"[PAIR_CTX] common neighbors: {' | '.join(common_text)}")

    # Relation motifs from 2-hop paths, without middle entity names by default.
    paths = find_two_hop_paths(head, tail, graph_info, max_paths=max_pair_motifs)
    motifs: List[str] = []
    seen = set()
    for r1, d1, _middle, r2, d2 in paths:
        r1_text = normalize_kg_text(r1, dataset_name, relation_label_map) if normalize_relation_text else r1
        r2_text = normalize_kg_text(r2, dataset_name, relation_label_map) if normalize_relation_text else r2
        motif = f"{d1}:{r1_text}->{d2}:{r2_text}"
        if motif not in seen:
            seen.add(motif)
            motifs.append(motif)
    if motifs:
        items.append(f"[PAIR_CTX] motifs: {' | '.join(motifs[:max_pair_motifs])}")

    # If the same (head, tail) pair has other train relations, they can be useful.
    # Exclude the current target relation to avoid relation-label leakage.
    known_pair_rels = [r for r in graph_info.get("pair_to_relations", {}).get((head, tail), []) if r != exclude_relation]
    if known_pair_rels:
        rel_texts = [normalize_kg_text(r, dataset_name, relation_label_map) if normalize_relation_text else r for r in known_pair_rels]
        items.append(f"[PAIR_CTX] other train pair relations: {' | '.join(rel_texts[:max_pair_motifs])}")

    return items


# ==================== MASKS / PRIORS ====================

def build_relation_mask_by_type_pair(
    type_pair_to_relations: Mapping[Tuple[str, str], Sequence[str]],
    relation_to_idx: Mapping[str, int],
) -> Dict[Tuple[str, str], List[int]]:
    mask = {}
    for type_pair, relations in type_pair_to_relations.items():
        idxs = sorted(relation_to_idx[rel] for rel in relations if rel in relation_to_idx)
        if idxs:
            mask[type_pair] = idxs
    return mask


def build_schema_prior_counts(
    graph_info: Mapping[str, object],
    relation_to_idx: Mapping[str, int],
) -> Dict[Tuple[str, str], Dict[int, int]]:
    """Convert raw relation-count schema prior to relation-index counts."""
    result: Dict[Tuple[str, str], Dict[int, int]] = {}
    for schema_pair, rel_counts in graph_info.get("schema_pair_relation_counts", {}).items():
        idx_counts = {}
        for rel, count in rel_counts.items():
            if rel in relation_to_idx:
                idx_counts[relation_to_idx[rel]] = int(count)
        if idx_counts:
            result[schema_pair] = idx_counts
    return result


# ==================== DATASET ====================

class KGRelationDataset(Dataset):
    """Dataset for relation prediction: predict relation in (head, ?, tail)."""

    def __init__(
        self,
        triplets: pd.DataFrame,
        tokenizer,
        relation_to_idx: Mapping[str, int],
        graph_info: Mapping[str, object],
        entity_types: Optional[Mapping[str, str]] = None,
        entity_label_map: Optional[Mapping[str, str]] = None,
        relation_label_map: Optional[Mapping[str, str]] = None,
        dataset_name: str = "",
        max_length: int = 128,
        pretokenize: bool = False,
        dynamic_padding: bool = True,
        default_entity_type: str = "UNK",
        normalize_entity_text: bool = True,
        normalize_relation_text: bool = True,
        context_mode: str = "adaptive",
        min_context: int = 5,
        base_context: int = 15,
        max_context: int = 30,
        max_head_context_tokens: int = 40,
        max_tail_context_tokens: int = 40,
        max_pair_context_tokens: int = 24,
        max_path_context_tokens: int = 20,
        use_second_hop_context: bool = False,
        second_hop_budget: int = 5,
        use_entity_types: bool = False,
        use_schema_tokens: bool = True,
        use_pair_context: bool = True,
        max_common_neighbors: int = 5,
        max_pair_motifs: int = 5,
        use_path_context: bool = False,
        path_context_mode: str = "motif",
        max_paths: int = 3,
        group_context_by_relation: bool = True,
        max_neighbors_per_relation: int = 3,
        w_degree: float = 1.0,
        w_rel_rarity: float = 0.35,
        w_redundancy: float = 0.15,
        training: bool = False,
        use_context_dropout: bool = False,
        context_dropout_prob: float = 0.0,
        shuffle_context_items: bool = False,
    ):
        self.triplets = triplets.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.relation_to_idx = dict(relation_to_idx)
        self.graph_info = graph_info
        self.entity_types = entity_types or {}
        self.entity_label_map = entity_label_map or {}
        self.relation_label_map = relation_label_map or {}
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.pretokenize = pretokenize
        self.dynamic_padding = dynamic_padding
        self.default_entity_type = default_entity_type
        self.normalize_entity_text = normalize_entity_text
        self.normalize_relation_text = normalize_relation_text
        self.context_mode = context_mode
        self.min_context = min_context
        self.base_context = base_context
        self.max_context = max_context
        self.max_head_context_tokens = max_head_context_tokens
        self.max_tail_context_tokens = max_tail_context_tokens
        self.max_pair_context_tokens = max_pair_context_tokens
        self.max_path_context_tokens = max_path_context_tokens
        self.use_second_hop_context = use_second_hop_context
        self.second_hop_budget = second_hop_budget
        self.use_entity_types = use_entity_types
        self.use_schema_tokens = use_schema_tokens
        self.use_pair_context = use_pair_context
        self.max_common_neighbors = max_common_neighbors
        self.max_pair_motifs = max_pair_motifs
        self.use_path_context = use_path_context
        self.path_context_mode = path_context_mode
        self.max_paths = max_paths
        self.group_context_by_relation = group_context_by_relation
        self.max_neighbors_per_relation = max_neighbors_per_relation
        self.w_degree = w_degree
        self.w_rel_rarity = w_rel_rarity
        self.w_redundancy = w_redundancy
        self.training = training
        self.use_context_dropout = use_context_dropout
        self.context_dropout_prob = context_dropout_prob
        self.shuffle_context_items = shuffle_context_items

        self.texts: List[str] = []
        self.labels: List[int] = []
        self.metas: List[Dict[str, object]] = []
        for row in self.triplets.itertuples(index=False):
            head, relation, tail = str(row.head), str(row.relation), str(row.tail)
            if relation not in self.relation_to_idx:
                raise KeyError(f"Relation {relation!r} is not in relation_to_idx.")
            text, meta = self._build_text_and_meta(head, relation, tail, augment=False)
            self.texts.append(text)
            self.labels.append(self.relation_to_idx[relation])
            self.metas.append(meta)

        self.labels_tensor = torch.tensor(self.labels, dtype=torch.long)
        self.encodings = None
        if self.pretokenize:
            self.encodings = self.tokenizer(
                self.texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

    def _maybe_augment_items(self, items: List[str]) -> List[str]:
        if not self.training:
            return items
        out = list(items)
        if self.use_context_dropout and self.context_dropout_prob > 0:
            kept = [x for x in out if random.random() >= self.context_dropout_prob]
            out = kept or out[:1]
        if self.shuffle_context_items:
            random.shuffle(out)
        return out

    def _build_text_and_meta(self, head: str, relation: str, tail: str, augment: bool = False) -> Tuple[str, Dict[str, object]]:
        head_type = get_entity_type(head, self.entity_types, self.default_entity_type)
        tail_type = get_entity_type(tail, self.entity_types, self.default_entity_type)
        head_schema = self.graph_info.get("entity_schema", {}).get(head, "UNK_SCHEMA")
        tail_schema = self.graph_info.get("entity_schema", {}).get(tail, "UNK_SCHEMA")
        exclude_triple = (head, relation, tail)

        h_items = build_entity_context_items(
            head,
            role="head",
            graph_info=self.graph_info,
            entity_types=self.entity_types,
            dataset_name=self.dataset_name,
            entity_label_map=self.entity_label_map,
            relation_label_map=self.relation_label_map,
            normalize_entity_text=self.normalize_entity_text,
            normalize_relation_text=self.normalize_relation_text,
            default_type=self.default_entity_type,
            context_mode=self.context_mode,
            min_context=self.min_context,
            base_context=self.base_context,
            max_context=self.max_context,
            use_second_hop=self.use_second_hop_context,
            second_hop_budget=self.second_hop_budget,
            w_degree=self.w_degree,
            w_rel_rarity=self.w_rel_rarity,
            w_redundancy=self.w_redundancy,
            exclude_triple=exclude_triple,
            group_context_by_relation=self.group_context_by_relation,
            max_neighbors_per_relation=self.max_neighbors_per_relation,
        )
        t_items = build_entity_context_items(
            tail,
            role="tail",
            graph_info=self.graph_info,
            entity_types=self.entity_types,
            dataset_name=self.dataset_name,
            entity_label_map=self.entity_label_map,
            relation_label_map=self.relation_label_map,
            normalize_entity_text=self.normalize_entity_text,
            normalize_relation_text=self.normalize_relation_text,
            default_type=self.default_entity_type,
            context_mode=self.context_mode,
            min_context=self.min_context,
            base_context=self.base_context,
            max_context=self.max_context,
            use_second_hop=self.use_second_hop_context,
            second_hop_budget=self.second_hop_budget,
            w_degree=self.w_degree,
            w_rel_rarity=self.w_rel_rarity,
            w_redundancy=self.w_redundancy,
            exclude_triple=exclude_triple,
            group_context_by_relation=self.group_context_by_relation,
            max_neighbors_per_relation=self.max_neighbors_per_relation,
        )

        if augment:
            h_items = self._maybe_augment_items(h_items)
            t_items = self._maybe_augment_items(t_items)

        h_items = pack_context_items(self.tokenizer, h_items, self.max_head_context_tokens)
        t_items = pack_context_items(self.tokenizer, t_items, self.max_tail_context_tokens)

        pair_items: List[str] = []
        if self.use_pair_context:
            pair_items = build_pair_context_items(
                head,
                tail,
                self.graph_info,
                dataset_name=self.dataset_name,
                entity_label_map=self.entity_label_map,
                relation_label_map=self.relation_label_map,
                normalize_entity_text=self.normalize_entity_text,
                normalize_relation_text=self.normalize_relation_text,
                max_common_neighbors=self.max_common_neighbors,
                max_pair_motifs=self.max_pair_motifs,
                exclude_relation=relation,
            )
            if augment:
                pair_items = self._maybe_augment_items(pair_items)
            pair_items = pack_context_items(self.tokenizer, pair_items, self.max_pair_context_tokens)

        path_items: List[str] = []
        if self.use_path_context:
            paths = find_two_hop_paths(head, tail, self.graph_info, max_paths=self.max_paths)
            path_items = path_items_to_text(
                head,
                tail,
                paths,
                dataset_name=self.dataset_name,
                entity_label_map=self.entity_label_map,
                relation_label_map=self.relation_label_map,
                normalize_entity_text=self.normalize_entity_text,
                normalize_relation_text=self.normalize_relation_text,
                mode=self.path_context_mode,
            )
            if augment:
                path_items = self._maybe_augment_items(path_items)
            path_items = pack_context_items(self.tokenizer, path_items, self.max_path_context_tokens)

        h_text = normalize_kg_text(head, self.dataset_name, self.entity_label_map) if self.normalize_entity_text else head
        t_text = normalize_kg_text(tail, self.dataset_name, self.entity_label_map) if self.normalize_entity_text else tail
        head_type_text = normalize_kg_text(head_type, self.dataset_name, None) if self.normalize_entity_text else head_type
        tail_type_text = normalize_kg_text(tail_type, self.dataset_name, None) if self.normalize_entity_text else tail_type
        head_schema_text = normalize_kg_text(head_schema, self.dataset_name, None)
        tail_schema_text = normalize_kg_text(tail_schema, self.dataset_name, None)

        parts = []
        if self.use_entity_types:
            parts.append(f"[HEAD_TYPE] {head_type_text}")
        if self.use_schema_tokens:
            parts.append(f"[HEAD_SCHEMA] {head_schema_text}")
        parts.append(f"[HEAD] {h_text}")
        parts.append(f"[HEAD_CTX] {' '.join(h_items)}")
        if self.use_entity_types:
            parts.append(f"[TAIL_TYPE] {tail_type_text}")
        if self.use_schema_tokens:
            parts.append(f"[TAIL_SCHEMA] {tail_schema_text}")
        parts.append(f"[TAIL] {t_text}")
        parts.append(f"[TAIL_CTX] {' '.join(t_items)}")
        if pair_items:
            parts.append(" ".join(pair_items))
        if path_items:
            parts.append(" ".join(path_items))
        text = " ".join(parts)

        meta = {
            "head": head,
            "relation": relation,
            "tail": tail,
            "head_type": head_type,
            "tail_type": tail_type,
            "head_schema": head_schema,
            "tail_schema": tail_schema,
            "num_head_context_items": len(h_items),
            "num_tail_context_items": len(t_items),
            "num_pair_context_items": len(pair_items),
            "num_path_items": len(path_items),
            "text_token_count": len(self.tokenizer.tokenize(text)),
        }
        return text, meta

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        # If training augmentation is enabled, rebuild text dynamically for this
        # sample. Otherwise use cached text/meta.
        if self.training and (self.use_context_dropout or self.shuffle_context_items):
            row = self.triplets.iloc[idx]
            text, meta = self._build_text_and_meta(str(row.head), str(row.relation), str(row.tail), augment=True)
        else:
            text = self.texts[idx]
            meta = self.metas[idx]

        if self.encodings is not None:
            inputs = {key: val[idx] for key, val in self.encodings.items()}
        else:
            inputs = self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
        return inputs, self.labels_tensor[idx], meta


# ==================== COLLATION / DATALOADERS ====================

def make_relation_collate_fn(tokenizer, dynamic_padding: bool = True, max_length: int = 128):
    """Create a collate function that supports variable-length tokenization."""

    def collate(batch):
        inputs, labels, metas = zip(*batch)
        # Fixed pretokenized tensors arrive as tensors; variable encodings arrive as lists.
        if isinstance(inputs[0].get("input_ids"), torch.Tensor):
            batch_inputs = {key: torch.stack([x[key] for x in inputs], dim=0) for key in inputs[0].keys()}
        else:
            batch_inputs = tokenizer.pad(
                list(inputs),
                padding=True if dynamic_padding else "max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        labels_tensor = torch.stack(list(labels), dim=0) if torch.is_tensor(labels[0]) else torch.tensor(labels, dtype=torch.long)
        return batch_inputs, labels_tensor, list(metas)

    return collate


def create_dataloaders(
    train_triplets: pd.DataFrame,
    valid_triplets: pd.DataFrame,
    test_triplets: pd.DataFrame,
    tokenizer,
    relation_to_idx: Mapping[str, int],
    graph_info: Mapping[str, object],
    entity_types: Optional[Mapping[str, str]] = None,
    entity_label_map: Optional[Mapping[str, str]] = None,
    relation_label_map: Optional[Mapping[str, str]] = None,
    dataset_name: str = "",
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0,
    dynamic_padding: bool = True,
    **dataset_kwargs,
):
    """Create standard non-DDP dataloaders."""
    train_dataset = KGRelationDataset(
        train_triplets, tokenizer, relation_to_idx, graph_info, entity_types,
        entity_label_map=entity_label_map,
        relation_label_map=relation_label_map,
        dataset_name=dataset_name,
        max_length=max_length,
        dynamic_padding=dynamic_padding,
        training=True,
        **dataset_kwargs,
    )
    valid_dataset = KGRelationDataset(
        valid_triplets, tokenizer, relation_to_idx, graph_info, entity_types,
        entity_label_map=entity_label_map,
        relation_label_map=relation_label_map,
        dataset_name=dataset_name,
        max_length=max_length,
        dynamic_padding=dynamic_padding,
        training=False,
        **dataset_kwargs,
    )
    test_dataset = KGRelationDataset(
        test_triplets, tokenizer, relation_to_idx, graph_info, entity_types,
        entity_label_map=entity_label_map,
        relation_label_map=relation_label_map,
        dataset_name=dataset_name,
        max_length=max_length,
        dynamic_padding=dynamic_padding,
        training=False,
        **dataset_kwargs,
    )

    collate_fn = make_relation_collate_fn(tokenizer, dynamic_padding=dynamic_padding, max_length=max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_dataloader, valid_dataloader, test_dataloader
