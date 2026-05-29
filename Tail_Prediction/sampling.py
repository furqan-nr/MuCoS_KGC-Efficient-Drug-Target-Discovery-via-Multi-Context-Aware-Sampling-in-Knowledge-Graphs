from collections import Counter
import math

def _type_compatibility(entity, query_relation, entity_types=None):
    if not entity_types:
        return 0.0
    return 0.0


def _sort_key(score, head, relation, tail):
    return (-score, relation, head, tail)


def score_hc_candidates(candidates, query_relation, degrees, entity_types=None):
    scored = []
    for head, relation, tail in candidates:
        score = (
            0.60 * math.log1p(degrees.get(tail, 0))
            + 0.30 * (1.0 if relation == query_relation else 0.0)
            + 0.10 * _type_compatibility(tail, query_relation, entity_types)
        )
        scored.append((score, head, relation, tail))
    return sorted(scored, key=lambda x: _sort_key(x[0], x[1], x[2], x[3]))


def score_rc_candidates(candidates, query_relation, degrees, entity_types=None):
    scored = []
    for head, relation, tail in candidates:
        score = (
            0.45 * math.log1p(degrees.get(head, 0))
            + 0.45 * math.log1p(degrees.get(tail, 0))
            + 0.10 * _type_compatibility(tail, query_relation, entity_types)
        )
        scored.append((score, head, relation, tail))
    return sorted(scored, key=lambda x: _sort_key(x[0], x[1], x[2], x[3]))


def select_head_context(scored_candidates, budget, max_same_relation):
    selected = []
    relation_counts = Counter()
    deferred = []

    for score, head, relation, tail in scored_candidates:
        if relation_counts[relation] < max_same_relation:
            selected.append((head, relation, tail))
            relation_counts[relation] += 1
            if len(selected) >= budget:
                return selected
        else:
            deferred.append((head, relation, tail))

    if len(selected) < budget:
        remaining = budget - len(selected)
        selected.extend(deferred[:remaining])

    return selected


def select_relation_context(scored_candidates, budget):
    selected = []
    used_heads = set()
    used_tails = set()
    deferred = []

    for score, head, relation, tail in scored_candidates:
        if head not in used_heads and tail not in used_tails:
            selected.append((head, relation, tail))
            used_heads.add(head)
            used_tails.add(tail)
            if len(selected) >= budget:
                return selected
        else:
            deferred.append((head, relation, tail))

    if len(selected) < budget:
        remaining = budget - len(selected)
        selected.extend(deferred[:remaining])

    return selected


def budget_contexts(
    hc_candidates,
    rc_candidates,
    degrees,
    query_relation,
    max_hc,
    max_rc,
    max_total_context,
    max_rc_if_hc_short,
    max_same_relation_in_hc,
    entity_types=None,
):
    """Select head/relation contexts using the fixed MuCoS budgets."""
    hc_budget = min(max_hc, len(hc_candidates))

    unused_hc_slots = max_hc - hc_budget

    rc_budget = min(
        max_rc + unused_hc_slots,
        max_rc_if_hc_short,
        len(rc_candidates),
    )
    rc_budget = min(rc_budget, max_total_context - hc_budget)

    scored_hc = score_hc_candidates(hc_candidates, query_relation, degrees, entity_types)
    scored_rc = score_rc_candidates(rc_candidates, query_relation, degrees, entity_types)

    hc_selected = select_head_context(scored_hc, hc_budget, max_same_relation_in_hc)
    rc_selected = select_relation_context(scored_rc, rc_budget)

    return hc_selected, rc_selected
