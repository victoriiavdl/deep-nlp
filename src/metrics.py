"""Evaluation utilities shared across all models."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.config import ID2LABEL, LABEL_NAMES, RESULTS_DIR


def evaluate(y_true, y_pred, label_names=None):
    """Return a dict with accuracy, macro-F1, weighted-F1, and per-class report."""
    label_names = label_names or LABEL_NAMES
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_report(metrics: dict, model_name: str = ""):
    """Pretty-print a metrics dict."""
    header = f"  {model_name}  " if model_name else ""
    print(f"\n{'='*60}")
    if header:
        print(f"{header:^60}")
        print(f"{'='*60}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1     : {metrics['f1_macro']:.4f}")
    print(f"  Weighted-F1  : {metrics['f1_weighted']:.4f}")
    print(f"{'='*60}")
    # per-class
    report = metrics["classification_report"]
    for label in LABEL_NAMES:
        if label in report:
            r = report[label]
            print(f"  {label:10s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  n={r['support']}")
    print()


# ── Leaderboard I/O ───────────────────────────────────────────────────

_LEADERBOARD_FILE = RESULTS_DIR / "leaderboard.json"


def _load_leaderboard() -> list[dict]:
    if _LEADERBOARD_FILE.exists():
        return json.loads(_LEADERBOARD_FILE.read_text())
    return []


def save_result(model_name: str, metrics: dict, extra: dict | None = None):
    """Append or update a model entry in the leaderboard JSON."""
    entries = _load_leaderboard()
    entry = {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
    }
    if extra:
        entry.update(extra)
    # upsert
    entries = [e for e in entries if e["model"] != model_name]
    entries.append(entry)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _LEADERBOARD_FILE.write_text(json.dumps(entries, indent=2))


def build_leaderboard():
    """Read leaderboard.json → produce sorted CSV and Markdown table."""
    entries = _load_leaderboard()
    if not entries:
        print("No results found. Run training scripts first.")
        return

    df = pd.DataFrame(entries)
    sort_col = "f1_macro"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"

    csv_path = RESULTS_DIR / "leaderboard.csv"
    md_path = RESULTS_DIR / "leaderboard.md"

    df.to_csv(csv_path)
    print(f"Saved {csv_path}")

    # Markdown
    cols = [c for c in df.columns if c != "classification_report"]
    md = df[cols].to_markdown()
    md_path.write_text(f"# Model Leaderboard\n\nSorted by **macro-F1** (descending).\n\n{md}\n")
    print(f"Saved {md_path}")

    # Console
    print(f"\n{md}\n")
    return df
