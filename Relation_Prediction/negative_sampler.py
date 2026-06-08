import os
import random
import torch


class NegativeSampler:
    """Simple per-entity negative pool cache and sampler.

    This is a lightweight, local implementation intended as a reusable
    utility for hard-negative mining experiments. It creates a per-entity
    pool of candidate negative entities (strings) and supports sampling.
    The pool is saved to disk so expensive construction happens once.
    """

    def __init__(self, model_save_path, pool_size: int = 100, seed: int = 42):
        self.model_save_path = model_save_path
        self.pool_size = pool_size
        self.rng = random.Random(seed)
        self.entity_pools = {}  # entity_str -> list(entity_str)

    def build_from_triplets(self, triplets_df, entity_incoming_neighbors=None, pool_size=None):
        if pool_size is None:
            pool_size = self.pool_size

        heads = set(triplets_df['head'].tolist())
        tails = set(triplets_df['tail'].tolist())
        all_entities = list(heads.union(tails))
        entity_set = set(all_entities)

        # Build for every entity a random pool avoiding self and direct positives
        for ent in all_entities:
            excluded = set()
            excluded.add(ent)
            if entity_incoming_neighbors and ent in entity_incoming_neighbors:
                # exclude known positive neighbors to avoid trivial negatives
                excluded.update(entity_incoming_neighbors.get(ent, []))

            candidates = [e for e in all_entities if e not in excluded]
            if len(candidates) <= pool_size:
                pool = list(candidates)
                self.rng.shuffle(pool)
            else:
                pool = self.rng.sample(candidates, pool_size)

            self.entity_pools[ent] = pool

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.entity_pools, path)

    def load(self, path: str):
        self.entity_pools = torch.load(path)

    def sample(self, entity, k=1):
        pool = self.entity_pools.get(entity)
        if not pool:
            return []
        if k >= len(pool):
            return list(pool)
        return self.rng.sample(pool, k)

    def batch_sample(self, entities, k=1):
        return [self.sample(e, k) for e in entities]
