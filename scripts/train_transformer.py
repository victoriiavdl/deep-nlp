#!/usr/bin/env python
"""Train a transformer model (DistilBERT, BERT, or FinBERT) for sentiment classification.

Usage:
    python scripts/train_transformer.py --model FinBERT
    python scripts/train_transformer.py --model DistilBERT
    python scripts/train_transformer.py --model BERT
    python scripts/train_transformer.py --all          # train all three

GPU Memory Optimization (8 GB VRAM, 85% budget = ~6.9 GB):
    - fp16 mixed precision for all models
    - gradient checkpointing for BERT-base and FinBERT (saves ~40% activation memory)
    - gradient accumulation (effective batch = per_device_batch * accum_steps)
    - PYTORCH_CUDA_ALLOC_CONF for reduced fragmentation
    - explicit torch.cuda.empty_cache() between models
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    MAX_LENGTH,
    MODELS_DIR,
    NUM_CLASSES,
    NUM_EPOCHS,
    RESULTS_DIR,
    SEED,
    TRANSFORMER_MODELS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from src.data import get_splits
from src.metrics import evaluate, print_report, save_result

# ── GPU memory optimization ───────────────────────────────────────────
# Limit PyTorch to 85% of VRAM to leave headroom for OS/display
GPU_MEMORY_FRACTION = 0.85

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

# Per-model tuning: (per_device_batch, gradient_accum_steps, gradient_checkpointing)
# Effective batch size = per_device_batch * accum_steps
MODEL_GPU_CONFIG = {
    "DistilBERT": {"batch_size": 32, "accum_steps": 1, "grad_ckpt": False},   # ~270MB model, fits easily
    "BERT":       {"batch_size": 16, "accum_steps": 2, "grad_ckpt": True},    # ~440MB model, needs ckpt
    "FinBERT":    {"batch_size": 16, "accum_steps": 2, "grad_ckpt": True},    # ~440MB model, needs ckpt
}


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        budget_mb = total_mb * GPU_MEMORY_FRACTION
        print(f"[device] CUDA GPU: {name}")
        print(f"[device] VRAM: {total_mb:.0f} MB total, budget: {budget_mb:.0f} MB ({GPU_MEMORY_FRACTION:.0%})")
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, 0)
        return "cuda"
    else:
        print("[device] ERROR: No CUDA GPU detected.")
        print("         Transformer fine-tuning requires a GPU.")
        print("         Please ensure CUDA is available and try again.")
        sys.exit(1)


def free_gpu_memory():
    """Force-free GPU memory between model trainings."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def train_model(model_key: str, device: str):
    """Train a single transformer model and save results."""
    hf_name = TRANSFORMER_MODELS[model_key]
    gpu_cfg = MODEL_GPU_CONFIG[model_key]

    print(f"\n{'='*60}")
    print(f"  Training: {model_key} ({hf_name})")
    print(f"{'='*60}")
    print(f"  GPU config: batch={gpu_cfg['batch_size']}, accum={gpu_cfg['accum_steps']}, "
          f"grad_ckpt={gpu_cfg['grad_ckpt']}")
    print(f"  Effective batch size: {gpu_cfg['batch_size'] * gpu_cfg['accum_steps']}")

    free_gpu_memory()

    # ── Data ───────────────────────────────────────────────────────
    train_df, val_df, test_df = get_splits()
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {split:5s}: {len(df)} samples")

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df[["headline", "label"]].reset_index(drop=True)),
        "val": Dataset.from_pandas(val_df[["headline", "label"]].reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df[["headline", "label"]].reset_index(drop=True)),
    })

    # ── Tokenize ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    def tokenize_fn(batch):
        return tokenizer(batch["headline"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # ── Model ──────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    if gpu_cfg["grad_ckpt"]:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: ENABLED (saves ~40% activation memory)")

    n_params = count_parameters(model)
    print(f"  Trainable parameters: {n_params:,}")

    # ── Training ───────────────────────────────────────────────────
    output_dir = RESULTS_DIR / f"{model_key}_training"
    save_dir = MODELS_DIR / model_key

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=gpu_cfg["batch_size"],
        per_device_eval_batch_size=gpu_cfg["batch_size"] * 2,
        gradient_accumulation_steps=gpu_cfg["accum_steps"],
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=25,
        fp16=True,
        seed=SEED,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=0,    # avoid multiprocessing issues on Windows
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Log peak memory before training
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU memory: {peak_mb:.0f} MB")

    # ── Evaluate on test set ───────────────────────────────────────
    t0 = time.time()
    preds_output = trainer.predict(tokenized["test"])
    infer_time = time.time() - t0

    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = preds_output.label_ids

    metrics = evaluate(y_true, y_pred)
    print_report(metrics, model_key)

    peak_mb_str = f"{peak_mb:.0f} MB" if device == "cuda" else "N/A"
    save_result(model_key, metrics, extra={
        "type": "transformer",
        "hf_model": hf_name,
        "train_time_s": round(train_time, 1),
        "inference_time_s": round(infer_time, 4),
        "params": f"{n_params:,}",
        "peak_gpu_mb": peak_mb_str,
    })

    # ── Save model ─────────────────────────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"  Model saved to {save_dir}")

    # Cleanup before next model
    del model, trainer
    free_gpu_memory()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train transformer models for sentiment analysis")
    parser.add_argument("--model", type=str, choices=list(TRANSFORMER_MODELS.keys()),
                        help="Which model to train")
    parser.add_argument("--all", action="store_true", help="Train all transformer models")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.print_help()
        print("\nExample: python scripts/train_transformer.py --model FinBERT")
        sys.exit(1)

    device = get_device()

    if args.all:
        models_to_train = list(TRANSFORMER_MODELS.keys())
    else:
        models_to_train = [args.model]

    for model_key in models_to_train:
        train_model(model_key, device)

    print("\nAll transformer training complete. Results in results/leaderboard.json")


if __name__ == "__main__":
    main()
