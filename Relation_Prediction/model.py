"""Model/tokenizer loader for general KG relation prediction."""

from typing import Iterable, Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    special_tokens: Optional[Iterable[str]] = None,
) -> Tuple[object, object]:
    """Load a HuggingFace sequence-classification model and tokenizer.

    Auto classes make it possible to test DistilBERT, BERT, RoBERTa,
    DeBERTa, or other encoder-only classifiers without changing train.py.
    The method remains negative-sample-free: the classifier predicts the
    relation label directly from the encoded (head, tail, context) sequence.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    if special_tokens:
        special_tokens = list(dict.fromkeys(special_tokens))
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
