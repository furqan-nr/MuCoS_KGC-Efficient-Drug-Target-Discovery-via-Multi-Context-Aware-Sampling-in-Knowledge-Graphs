import torch
from torch.utils.data import Dataset

from utils import get_one_hop_head_entity_neighbors, get_one_hop_tail_entity_neighbors


def _join_neighbors(neighbors):
    if not neighbors:
        return ""
    return " ".join(neighbors)


class KGDataset(Dataset):
    def __init__(self, triplets, tokenizer, relation_to_idx, all_triplets):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.relation_to_idx = relation_to_idx
        self.all_triplets = all_triplets

        texts = []
        labels = []
        for row in self.triplets.itertuples(index=False):
            head, relation, tail = row.head, row.relation, row.tail

            head_neighbors = get_one_hop_head_entity_neighbors(head, self.all_triplets, max_degree=20)
            tail_neighbors = get_one_hop_tail_entity_neighbors(tail, self.all_triplets, max_degree=20)

            head_neighbors_str = _join_neighbors(head_neighbors)
            tail_neighbors_str = _join_neighbors(tail_neighbors)

            texts.append(f"{head} [SEP] {head_neighbors_str} [SEP] {tail} [SEP] {tail_neighbors_str}")
            labels.append(self.relation_to_idx[relation])

        self.encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        inputs = {key: val[idx] for key, val in self.encodings.items()}
        label = self.labels[idx]
        return inputs, label