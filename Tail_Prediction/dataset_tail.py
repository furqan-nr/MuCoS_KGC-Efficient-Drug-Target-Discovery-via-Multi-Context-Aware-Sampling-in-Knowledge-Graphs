import json
import os

import torch
from torch.utils.data import Dataset


class TailContextDataset(Dataset):
    """Streaming JSONL-backed dataset that indexes line byte offsets.

    This avoids loading the entire JSONL into memory. Each __getitem__ opens
    the file, seeks to the stored offset, reads one line, decodes and parses it.
    """

    def __init__(self, jsonl_path, tokenizer, max_length=128, tokenized_path=None, use_tokenized_cache=True):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_path = tokenized_path
        self.use_tokenized_cache = use_tokenized_cache

        self._tokenized = None
        if self.use_tokenized_cache and self.tokenized_path and os.path.exists(self.tokenized_path):
            self._tokenized = torch.load(self.tokenized_path, map_location="cpu")
            return

        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"JSONL not found: {self.jsonl_path}")

        # Build byte offsets index
        self._offsets = []
        with open(self.jsonl_path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                self._offsets.append(pos)
        # Per-worker file handle; will be opened lazily in each worker process
        self._fh = None

    def __len__(self):
        if self._tokenized is not None:
            return len(self._tokenized)
        return len(self._offsets)

    def __getitem__(self, idx):
        if self._tokenized is not None:
            record = self._tokenized[idx]
            inputs = {
                "input_ids": record["input_ids"],
                "attention_mask": record["attention_mask"],
            }
            label = record["label"]
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long)
            meta = record.get("metadata", {})
            return inputs, label, meta

        offset = self._offsets[idx]
        # Open a persistent file handle per worker/process for faster reads
        if self._fh is None:
            self._fh = open(self.jsonl_path, "rb")
        f = self._fh
        f.seek(offset)
        raw = f.readline()
        try:
            record = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSONL at {self.jsonl_path} offset {offset}: {e}")

        inputs = self.tokenizer(
            record["input_text"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        label = torch.tensor(record["label"], dtype=torch.long)
        meta = {
            "head": record["head"],
            "relation": record["relation"],
            "tail": record["tail"],
        }
        return inputs, label, meta

    def close(self):
        if getattr(self, "_fh", None) is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def __del__(self):
        # Ensure filehandle closed on dataset destruction
        self.close()