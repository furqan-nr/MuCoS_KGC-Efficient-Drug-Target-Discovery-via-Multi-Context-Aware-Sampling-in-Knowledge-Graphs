import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TAIL_DIR = ROOT / "Tail_Prediction"
if str(TAIL_DIR) not in sys.path:
    sys.path.insert(0, str(TAIL_DIR))

import config_tail
import preprocess_contexts
from dataset_tail import TailContextDataset


class FakeTokenizer:
    def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None, add_special_tokens=None):
        token_count = min(len(text.split()), max_length or len(text.split()))
        input_ids = list(range(1, token_count + 1))
        attention_mask = [1] * token_count
        if max_length is not None:
            pad_length = max_length - token_count
            input_ids = input_ids + [0] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        result = {
            "input_ids": __import__("torch").tensor([input_ids], dtype=__import__("torch").long),
            "attention_mask": __import__("torch").tensor([attention_mask], dtype=__import__("torch").long),
        }
        return result


class TestTailPredictionPipeline(unittest.TestCase):
    def setUp(self):
        self._config_snapshot = {
            "DATA_DIR": config_tail.DATA_DIR,
            "train_file_path": config_tail.train_file_path,
            "valid_file_path": config_tail.valid_file_path,
            "test_file_path": config_tail.test_file_path,
            "PROCESSED_DIR": config_tail.PROCESSED_DIR,
            "OUTPUT_DIR": config_tail.OUTPUT_DIR,
            "MODEL_NAME": config_tail.MODEL_NAME,
            "MAX_LENGTH": config_tail.MAX_LENGTH,
            "MAX_HC": config_tail.MAX_HC,
            "MAX_RC": config_tail.MAX_RC,
            "MAX_TOTAL_CONTEXT": config_tail.MAX_TOTAL_CONTEXT,
            "MAX_RC_IF_HC_SHORT": config_tail.MAX_RC_IF_HC_SHORT,
            "MAX_SAME_RELATION_IN_HC": config_tail.MAX_SAME_RELATION_IN_HC,
        }
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    def tearDown(self):
        for key, value in self._config_snapshot.items():
            setattr(config_tail, key, value)

    def _write_split(self, name, rows):
        path = os.path.join(self.tempdir.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            for head, relation, tail in rows:
                handle.write(f"{head}\t{relation}\t{tail}\n")
        return path

    def test_preprocess_all_builds_train_only_contexts(self):
        train_path = self._write_split(
            "train.txt",
            [
                ("drug_a", "treats", "disease_x"),
                ("drug_a", "interacts", "protein_y"),
            ],
        )
        valid_path = self._write_split("valid.txt", [("drug_b", "treats", "disease_x")])
        test_path = self._write_split("test.txt", [("drug_a", "treats", "disease_z")])

        config_tail.train_file_path = train_path
        config_tail.valid_file_path = valid_path
        config_tail.test_file_path = test_path
        config_tail.PROCESSED_DIR = os.path.join(self.tempdir.name, "processed")
        config_tail.MODEL_NAME = "unused-model-name"
        config_tail.MAX_LENGTH = 32
        config_tail.MAX_HC = 15
        config_tail.MAX_RC = 5
        config_tail.MAX_TOTAL_CONTEXT = 20
        config_tail.MAX_RC_IF_HC_SHORT = 10
        config_tail.MAX_SAME_RELATION_IN_HC = 5

        fake_tokenizer = FakeTokenizer()
        with patch.object(preprocess_contexts.DistilBertTokenizer, "from_pretrained", return_value=fake_tokenizer):
            result = preprocess_contexts.preprocess_all()

        self.assertTrue(os.path.exists(result["train_path"]))
        self.assertTrue(os.path.exists(result["valid_path"]))
        self.assertTrue(os.path.exists(result["test_path"]))

        with open(result["train_path"], "r", encoding="utf-8") as handle:
            train_records = [json.loads(line) for line in handle]

        self.assertEqual(len(train_records), 2)
        for record in train_records:
            triple_text = f"{record['head']}-{record['relation']}-{record['tail']}"
            self.assertNotIn(triple_text, record["head_context"])
            self.assertNotIn(triple_text, record["relation_context"])

        stats = result["dataset_stats"]
        self.assertEqual(stats["num_train_triples"], 2)
        self.assertEqual(stats["num_valid_triples"], 1)
        self.assertEqual(stats["num_test_triples"], 1)

    def test_streaming_dataset_reads_jsonl_records(self):
        jsonl_path = os.path.join(self.tempdir.name, "sample.jsonl")
        record = {
            "head": "drug_a",
            "relation": "treats",
            "tail": "disease_x",
            "input_text": "drug_a [SEP]  [SEP] treats [SEP] ",
            "label": 3,
        }
        with open(jsonl_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.write(json.dumps({**record, "tail": "disease_y", "label": 4}) + "\n")

        dataset = TailContextDataset(jsonl_path, FakeTokenizer(), max_length=8)

        self.assertEqual(len(dataset), 2)
        inputs, label, meta = dataset[0]
        self.assertEqual(label.item(), 3)
        self.assertEqual(meta["head"], "drug_a")
        self.assertEqual(meta["relation"], "treats")
        self.assertEqual(meta["tail"], "disease_x")
        self.assertIn("input_ids", inputs)
        self.assertEqual(inputs["input_ids"].shape[0], 8)


if __name__ == "__main__":
    unittest.main()