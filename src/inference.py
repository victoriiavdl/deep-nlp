"""Shared inference utility — used by evaluation scripts and app.py."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.config import ID2LABEL, LABEL2ID, MAX_LENGTH, MODELS_DIR, RESULTS_DIR


def get_device():
    """Return torch device, preferring CUDA."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[device] Using CPU")
    return dev


def load_best_model(model_dir=None):
    """Load the best fine-tuned transformer model + tokenizer.

    Tries FinBERT first, then BERT, then DistilBERT.
    Returns (model, tokenizer, model_name).
    """
    candidates = ["FinBERT", "BERT", "DistilBERT"]
    base = model_dir or MODELS_DIR

    for name in candidates:
        path = base / name
        if path.exists() and (path / "config.json").exists():
            print(f"[inference] Loading {name} from {path}")
            tokenizer = AutoTokenizer.from_pretrained(str(path))
            model = AutoModelForSequenceClassification.from_pretrained(str(path))
            return model, tokenizer, name

    raise FileNotFoundError(
        f"No fine-tuned model found in {base}. "
        "Run training scripts first (e.g. python scripts/train_transformer.py --model FinBERT)."
    )


def build_pipeline(model=None, tokenizer=None, device=None):
    """Build a HuggingFace text-classification pipeline."""
    if model is None or tokenizer is None:
        model, tokenizer, _ = load_best_model()
    if device is None:
        device = get_device()
    device_id = 0 if device.type == "cuda" else -1
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
        top_k=None,  # return all scores
        max_length=MAX_LENGTH,
        truncation=True,
    )


def predict(text: str, pipe) -> dict:
    """Run inference on a single text. Returns {label: str, scores: dict}."""
    results = pipe(text[:512])[0]
    scores = {r["label"]: round(r["score"], 4) for r in results}
    best = max(results, key=lambda r: r["score"])
    return {"label": best["label"], "confidence": round(best["score"], 4), "scores": scores}
