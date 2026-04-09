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
│   ├���─ 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_lstm.ipynb
│   └── 04_finbert.ipynb
├── results/                       # Training outputs (auto-generated)
│   ├── leaderboard.json           # Machine-readable results
│   ├── leaderboard.csv            # Sorted table
│   └── leaderboard.md             # Markdown table
├── app.py                         # Gradio demo interface
├── setup.bat                      # One-click environment setup (Windows)
├── run.bat                        # Command runner (Windows)
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

### Quick Start (Windows)

The easiest way to set everything up is the provided batch script:

```
setup.bat
```

This automatically finds `uv` on your system (even if it is not in PATH), installs all dependencies including CUDA-enabled PyTorch, and prints the commands to run next.

If `uv` is not installed at all, the script falls back to creating a `.venv` with pip and gives you the extra step needed for GPU support.

### Manual Installation

**Option A — with [uv](https://docs.astral.sh/uv/) (recommended)**

If `uv` is in your PATH:

```
uv sync
```

If `uv` is installed but not in PATH (common on Windows), find it and call it directly. Typical locations:

```
%LOCALAPPDATA%\uv\uv.exe sync
%USERPROFILE%\.local\bin\uv.exe sync
%USERPROFILE%\.cargo\bin\uv.exe sync
```

The `pyproject.toml` is already configured to pull CUDA-enabled PyTorch from the correct index, so `uv sync` handles everything in one step.

**Option B — with pip**

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Important: pip installs CPU-only PyTorch by default. For GPU support, run this additional command after the install:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### GPU Verification

After setup, verify that PyTorch sees your GPU:

```
run.bat -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}')"
```

Or directly:

```
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Transformer training scripts check for CUDA at startup and **exit with a clear error** if no GPU is found, so you will not accidentally train on CPU.

## Running Scripts

All commands below use `run.bat`, which automatically finds `uv` or falls back to `.venv`. You can also substitute `run.bat` with `uv run python` or `.venv\Scripts\python` if you prefer.

### 1. Classical Baselines (CPU, ~10 seconds)

```
run.bat scripts/train_baselines.py
```

Trains TF-IDF + Logistic Regression and TF-IDF + Naive Bayes.

### 2. BiLSTM (CPU/GPU, ~1-2 minutes)

```
run.bat scripts/train_lstm.py
```

Bidirectional LSTM with early stopping. Runs on GPU if TensorFlow detects one, otherwise CPU.

### 3. Transformers (GPU required, ~2-5 min each)

```
:: Train a single model:
run.bat scripts/train_transformer.py --model FinBERT
run.bat scripts/train_transformer.py --model BERT
run.bat scripts/train_transformer.py --model DistilBERT

:: Or train all three:
run.bat scripts/train_transformer.py --all
```

Fine-tunes the chosen model for 5 epochs with early stopping (patience=2). Models are saved to `results/models/<ModelName>/`.

GPU memory is capped at 85% of available VRAM. On an 8 GB GPU, peak usage is around 2 GB per model thanks to fp16, gradient checkpointing, and gradient accumulation.

### 4. Build Leaderboard

```
run.bat scripts/build_leaderboard.py
```

Reads `results/leaderboard.json` and generates:
- `results/leaderboard.csv` — machine-readable
- `results/leaderboard.md` — formatted Markdown table
- Console output with the ranked table

### 5. Launch Demo

```
run.bat app.py
```

Opens a Gradio interface at `http://localhost:7860` with:
- Real-time sentiment prediction for any financial headline
- Confidence scores for all three classes
- Model leaderboard, dataset overview, and methodology tabs
- Example headlines to try

The app loads the best available fine-tuned model (FinBERT > BERT > DistilBERT).

## Results

### Model Leaderboard

| Rank | Model               | Accuracy | Macro-F1 | Type          | Train Time |
|------|---------------------|----------|----------|---------------|------------|
| 1    | FinBERT             | 88.1%    | 87.7%    | Transformer   | ~82s       |
| 2    | DistilBERT          | 86.8%    | 86.0%    | Transformer   | ~51s       |
| 3    | BERT                | 85.5%    | 83.8%    | Transformer   | ~111s      |
| 4    | TF-IDF + NaiveBayes | 71.3%    | 62.7%    | Classical ML  | <1s        |
| 5    | TF-IDF + LogReg     | 72.4%    | 60.1%    | Classical ML  | <1s        |
| 6    | BiLSTM              | 63.7%    | 37.3%    | Deep Learning | ~41s       |

### Model Comparison

| Type | Models | Strengths |
|------|--------|-----------|
| **Classical ML** | TF-IDF + LogReg, TF-IDF + NB | Fast, interpretable, strong baselines |
| **Deep Learning** | BiLSTM | Learns sequential patterns, no feature engineering |
| **Transformers** | DistilBERT, BERT, FinBERT | Pre-trained language understanding; FinBERT is domain-specialized for finance |

FinBERT, pre-trained on financial text, outperforms general-purpose models on this domain-specific task. The progression from bag-of-words to RNN to transformer illustrates the evolution of NLP techniques, with domain adaptation (FinBERT) providing a measurable edge over generic pre-training (BERT).

## Outputs

| Path | Git-tracked | Description |
|------|-------------|-------------|
| `results/leaderboard.json` | Yes | Raw benchmark results |
| `results/leaderboard.csv` | Yes | Sorted leaderboard table |
| `results/leaderboard.md` | Yes | Markdown leaderboard |
| `results/models/` | **No** | Saved model weights (~1 GB) |
| `results/*_training/` | **No** | Trainer checkpoints (~6 GB) |

Model weights and checkpoints are generated during training and excluded from git. To reproduce results on a new machine, run the training scripts.

## Configuration

All hyperparameters and paths are centralized in `src/config.py`. Key settings:

- `SEED = 42` — reproducibility
- `MAX_LENGTH = 128` — transformer tokenization
- `NUM_EPOCHS = 5` — transformer training
- `BATCH_SIZE = 16` — transformer batch size
- `LEARNING_RATE = 2e-5` — AdamW learning rate

## Notes

- The original Jupyter notebooks in `notebooks/` are preserved as reference and for interactive exploration.
- The root notebook `nlp-financial-news-sentiment-analysis.ipynb` is the original Kaggle notebook (reference only).
- The `FinancialPhraseBank/` directory contains the original dataset files at different agreement levels.
