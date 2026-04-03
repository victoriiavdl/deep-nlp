#!/usr/bin/env python
"""Train TF-IDF baselines: Logistic Regression and Naive Bayes."""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    LABEL_NAMES,
    RESULTS_DIR,
    SEED,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from src.data import clean_text, get_splits
from src.metrics import evaluate, print_report, save_result


def main():
    print("=" * 60)
    print("  TF-IDF Baseline Training")
    print("=" * 60)

    # ── Data ───────────────────────────────────────────────────────
    train_df, val_df, test_df = get_splits()
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {split_name:5s}: {len(split_df)} samples")

    train_df["clean"] = train_df["headline"].apply(clean_text)
    val_df["clean"] = val_df["headline"].apply(clean_text)
    test_df["clean"] = test_df["headline"].apply(clean_text)

    X_train, y_train = train_df["clean"].values, train_df["label"].values
    X_test, y_test = test_df["clean"].values, test_df["label"].values

    # ── TF-IDF ─────────────────────────────────────────────────────
    tfidf = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"\n  TF-IDF features: {X_train_tfidf.shape[1]}")

    # ── Logistic Regression ────────────────────────────────────────
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    t0 = time.time()
    lr.fit(X_train_tfidf, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred_lr = lr.predict(X_test_tfidf)
    infer_time = time.time() - t0

    metrics_lr = evaluate(y_test, y_pred_lr)
    print_report(metrics_lr, "TF-IDF + Logistic Regression")
    save_result("TF-IDF + LogReg", metrics_lr, extra={
        "type": "classical",
        "train_time_s": round(train_time, 2),
        "inference_time_s": round(infer_time, 4),
        "params": "TF-IDF features + LR",
    })

    # ── Naive Bayes ────────────────────────────────────────────────
    print("--- Naive Bayes ---")
    nb = MultinomialNB(alpha=0.1)
    t0 = time.time()
    nb.fit(X_train_tfidf, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred_nb = nb.predict(X_test_tfidf)
    infer_time = time.time() - t0

    metrics_nb = evaluate(y_test, y_pred_nb)
    print_report(metrics_nb, "TF-IDF + Naive Bayes")
    save_result("TF-IDF + NaiveBayes", metrics_nb, extra={
        "type": "classical",
        "train_time_s": round(train_time, 2),
        "inference_time_s": round(infer_time, 4),
        "params": "TF-IDF features + MultinomialNB",
    })

    print("Baseline results saved to results/leaderboard.json")


if __name__ == "__main__":
    main()
