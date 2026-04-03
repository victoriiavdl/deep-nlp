# Financial News Sentiment Analysis

Benchmarking classical ML, deep learning, and transformer models on the [FinancialPhraseBank](https://www.researchgate.net/publication/251231364_FinancialPhraseBank) dataset.

## Project Structure

```
├── data/
│   └── all-data.csv              # FinancialPhraseBank (4846 headlines, 3 classes)
├── src/                           # Reusable Python modules
│   ├── config.py                  # Central configuration & hyperparameters
│   ├── data.py                    # Data loading, cleaning, splitting
│   ├── metrics.py                 # Evaluation utilities & leaderboard I/O
│   └── inference.py               # Shared inference (used by scripts & app)
├── scripts/                       # Training & evaluation scripts
│   ├── train_baselines.py         # TF-IDF + Logistic Regression / Naive Bayes
│   ├── train_lstm.py              # Bidirectional LSTM (TensorFlow)
│   ├── train_transformer.py       # DistilBERT / BERT / FinBERT (PyTorch)
│   └── build_leaderboard.py       # Generate leaderboard from results
├── notebooks/                     # Original exploratory notebooks (reference)
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_lstm.ipynb
│   └── 04_finbert.ipynb
├── results/                       # Training outputs (auto-generated)
│   ├── leaderboard.json           # Machine-readable results
│   ├── leaderboard.csv            # Sorted table
│   └── leaderboard.md             # Markdown table
├── app.py                         # Gradio demo interface
├── pyproject.toml                 # Project metadata & dependencies
└── README.md
```

## Dataset

**FinancialPhraseBank** — 4,846 English financial news headlines annotated by domain experts.

| Class    | Count | Share  |
|----------|-------|--------|
| neutral  | 2,879 | 59.4%  |
| positive | 1,363 | 28.1%  |
| negative |   604 | 12.5%  |

The dataset is imbalanced, so **macro-F1** is the primary evaluation metric (alongside accuracy).

- Format: headerless CSV with columns `sentiment, headline`
- Encoding: `latin-1`
- Location: `data/all-data.csv`
- Split: 70% train / 10% validation / 20% test (stratified, seed=42)

## Setup

### Requirements

- Python 3.11+
- **GPU (CUDA)** required for transformer training (DistilBERT, BERT, FinBERT)
- CPU is sufficient for baselines and BiLSTM

### Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -e .
```

### GPU Verification

Transformer training scripts will check for CUDA and **exit with a clear error** if no GPU is found. This prevents accidentally running hours of training on CPU.

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Training

All scripts are run from the project root. Results are appended to `results/leaderboard.json`.

### 1. Classical Baselines (CPU, ~10 seconds)

```bash
uv run python scripts/train_baselines.py
```

Trains TF-IDF + Logistic Regression and TF-IDF + Naive Bayes.

### 2. BiLSTM (CPU/GPU, ~2-5 minutes)

```bash
uv run python scripts/train_lstm.py
```

Bidirectional LSTM with early stopping. Runs on GPU if TensorFlow detects one, otherwise CPU.

### 3. Transformers (GPU required, ~5-15 min each)

```bash
# Train a single model:
uv run python scripts/train_transformer.py --model FinBERT
uv run python scripts/train_transformer.py --model BERT
uv run python scripts/train_transformer.py --model DistilBERT

# Or train all three:
uv run python scripts/train_transformer.py --all
```

Fine-tunes the chosen model for 5 epochs with early stopping (patience=2). Models are saved to `results/models/<ModelName>/`.

## Leaderboard

After training, generate the sorted leaderboard:

```bash
uv run python scripts/build_leaderboard.py
```

This produces:
- `results/leaderboard.csv` — machine-readable
- `results/leaderboard.md` — formatted Markdown table
- Console output with the ranked table

### Model Comparison

| Type | Models | Strengths |
|------|--------|-----------|
| **Classical ML** | TF-IDF + LogReg, TF-IDF + NB | Fast, interpretable, strong baselines |
| **Deep Learning** | BiLSTM | Learns sequential patterns, no feature engineering |
| **Transformers** | DistilBERT, BERT, FinBERT | Pre-trained language understanding; FinBERT is domain-specialized for finance |

**Key insight**: FinBERT, pre-trained on financial text, is expected to outperform general-purpose models on this domain-specific task. The progression from bag-of-words → RNN → transformer tells a clear pedagogical story about NLP model evolution.

## Demo

Launch the interactive Gradio interface:

```bash
uv run python app.py
```

The app loads the best available fine-tuned model (FinBERT > BERT > DistilBERT) and provides:
- Real-time sentiment prediction for any financial headline
- Confidence scores for all three classes
- Example headlines to try

## Outputs

| Path | Git-tracked | Description |
|------|-------------|-------------|
| `results/leaderboard.json` | Yes | Raw benchmark results |
| `results/leaderboard.csv` | Yes | Sorted leaderboard table |
| `results/leaderboard.md` | Yes | Markdown leaderboard |
| `results/models/` | **No** | Saved model weights (heavy) |
| `results/*_training/` | **No** | Trainer checkpoints (heavy) |

## Configuration

All hyperparameters and paths are centralized in `src/config.py`. Key settings:

- `SEED = 42` — reproducibility
- `MAX_LENGTH = 128` — transformer tokenization
- `NUM_EPOCHS = 5` — transformer training
- `BATCH_SIZE = 16` — transformer batch size
- `LEARNING_RATE = 2e-5` — AdamW learning rate

## Notes

- The original Jupyter notebooks in `notebooks/` are preserved as reference and for interactive exploration.
- The root notebook `nlp-financial-news-sentiment-analysis.ipynb` is the original Kaggle notebook (reference only, contains deprecated APIs).
- The `FinancialPhraseBank/` directory contains the original dataset files at different agreement levels.
