#!/usr/bin/env python
"""Train a Bidirectional LSTM for sentiment classification."""

import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
    SpatialDropout1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    LABEL_NAMES,
    LSTM_BATCH_SIZE,
    LSTM_EMBED_DIM,
    LSTM_EPOCHS,
    LSTM_MAX_SEQ_LEN,
    LSTM_MAX_VOCAB,
    NUM_CLASSES,
    SEED,
)
from src.data import clean_text, get_splits
from src.metrics import evaluate, print_report, save_result


def build_model(vocab_size, embed_dim, seq_len, num_classes):
    model = Sequential([
        Embedding(vocab_size, embed_dim, input_length=seq_len, mask_zero=True),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)),
        Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.1)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def count_params(model):
    return int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))


def main():
    print("=" * 60)
    print("  Bidirectional LSTM Training")
    print("=" * 60)

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[device] TensorFlow GPU: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("[device] TensorFlow using CPU (LSTM training is feasible on CPU)")

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # ── Data ───────────────────────────────────────────────────────
    train_df, val_df, test_df = get_splits()
    train_df["clean"] = train_df["headline"].apply(lambda t: clean_text(t, lemmatize=False))
    val_df["clean"] = val_df["headline"].apply(lambda t: clean_text(t, lemmatize=False))
    test_df["clean"] = test_df["headline"].apply(lambda t: clean_text(t, lemmatize=False))

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {split_name:5s}: {len(split_df)} samples")

    # ── Tokenize & pad ─────────────────────────────────────────────
    tokenizer = Tokenizer(num_words=LSTM_MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean"].values)

    def to_padded(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=LSTM_MAX_SEQ_LEN, padding="post", truncating="post")

    X_train = to_padded(train_df["clean"].values)
    X_val = to_padded(val_df["clean"].values)
    X_test = to_padded(test_df["clean"].values)

    y_train = to_categorical(train_df["label"].values, NUM_CLASSES)
    y_val = to_categorical(val_df["label"].values, NUM_CLASSES)
    y_test_labels = test_df["label"].values

    # ── Build & train ──────────────────────────────────────────────
    model = build_model(LSTM_MAX_VOCAB, LSTM_EMBED_DIM, LSTM_MAX_SEQ_LEN, NUM_CLASSES)
    model.summary()
    n_params = count_params(model)
    print(f"\n  Trainable parameters: {n_params:,}")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - t0

    # ── Evaluate ───────────────────────────────────────────────────
    t0 = time.time()
    y_proba = model.predict(X_test, verbose=0)
    infer_time = time.time() - t0
    y_pred = np.argmax(y_proba, axis=1)

    metrics = evaluate(y_test_labels, y_pred)
    print_report(metrics, "Bidirectional LSTM")
    save_result("BiLSTM", metrics, extra={
        "type": "deep_learning",
        "train_time_s": round(train_time, 1),
        "inference_time_s": round(infer_time, 4),
        "params": f"{n_params:,}",
        "epochs_run": len(history.history["loss"]),
    })

    print("BiLSTM results saved to results/leaderboard.json")


if __name__ == "__main__":
    main()
