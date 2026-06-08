from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


def get_tokenizer_and_model_classes(model_name: str):
    """Returns tokenizer and model class (fixed to DistilBERT for this specific task)."""
    if "distilbert" in model_name.lower():
        return AutoTokenizer, AutoModelForSequenceClassification
    else:
        raise ValueError(f"Only DistilBERT is supported in this specific relation prediction version. Got: {model_name}")