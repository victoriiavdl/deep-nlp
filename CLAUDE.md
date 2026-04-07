# CLAUDE.md — Project Context for Claude Code

> **Self-update rule**: After completing any significant task (training, refactoring, adding features, fixing bugs), **re-edit this file** to reflect what changed. This keeps the context accurate across sessions. If you run out of context, this file is your single source of truth.

## Project Overview

**Financial News Sentiment Analysis** — benchmarking classical ML, deep learning, and transformer models on the FinancialPhraseBank dataset (4,846 financial headlines, 3 classes: positive/neutral/negative).

### Current State (last updated: 2026-04-07)

The project has been **fully refactored** from a notebook-only structure into a clean, modular, reproducible Python project. All 6 models have been trained and benchmarked. Two demo interfaces exist:

- **app.py** — Gradio demo (original, multi-tab, runs on port 7860)
- **streamlit_app.py** — Streamlit dashboard (victoria branch) with 8 tabs: Analyze, Batch Analysis, Live News, Leaderboard, Dataset, Methodology, Rendu, About

**New features (victoria branch, 2026-04-07):**
1. **Batch Analysis** — CSV upload → FinBERT prediction per row → pie chart + confidence histogram + CSV download
2. **Live News** — Yahoo Finance headlines via `yfinance` → real-time sentiment + overall sentiment gauge
3. **LIME Explainability** — Word importance highlighting (green/red) + bar chart after each prediction
4. **Web Speech API** — Browser-based voice input via `streamlit-javascript` (Chrome/Edge only)

## Repository Structure

```
deep-nlp/
├── data/all-data.csv              # Dataset (headerless CSV: sentiment,headline)
├── FinancialPhraseBank/           # Original dataset files (multiple agreement levels)
├── src/                           # Reusable Python modules
│   ├── __init__.py
│   ├── config.py                  # Central config: paths, hyperparams, model registry
│   ├── data.py                    # Data loading, cleaning, stratified 70/10/20 splitting
│   ├── metrics.py                 # Evaluation (accuracy, macro-F1, weighted-F1), leaderboard I/O
│   └── inference.py               # Shared inference: model loading, HF pipeline, predict()
├── scripts/
│   ├── train_baselines.py         # TF-IDF + LogReg + NaiveBayes (~10s on CPU)
│   ├── train_lstm.py              # BiLSTM via TensorFlow (~40s on CPU)
│   ├── train_transformer.py       # DistilBERT/BERT/FinBERT with GPU optimization (~1-2 min each)
│   └── build_leaderboard.py       # Aggregates results → CSV, JSON, Markdown
├── notebooks/                     # Original exploratory notebooks (preserved as reference)
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_lstm.ipynb
│   └── 04_finbert.ipynb
├── results/
│   ├── leaderboard.json           # Machine-readable results (git-tracked)
│   ├── leaderboard.csv            # Sorted leaderboard (git-tracked)
│   ├── leaderboard.md             # Markdown table (git-tracked)
│   ├── models/                    # Saved model weights (git-IGNORED, heavy)
│   └── *_training/                # Trainer checkpoints (git-IGNORED, heavy)
├── app.py                         # Gradio demo (loads best fine-tuned model)
├── streamlit_app.py               # Streamlit dashboard (batch, live news, LIME, voice)
├── pyproject.toml                 # Dependencies + uv CUDA PyTorch source
├── uv.lock                        # Lock file
├── .gitignore
└── README.md
```

## Key Decisions

- **Label mapping**: positive=0, neutral=1, negative=2 (consistent across all models)
- **Split**: 70/10/20 (train/val/test), stratified, seed=42
- **Primary metric**: macro-F1 (due to class imbalance: neutral 59%, positive 28%, negative 12%)
- **GPU**: RTX 5070 Laptop (8 GB VRAM), capped at 85% usage via `torch.cuda.set_per_process_memory_fraction(0.85)`
- **GPU optimizations**: fp16, gradient checkpointing for BERT/FinBERT, gradient accumulation, explicit `empty_cache()` between models
- **PyTorch CUDA**: Installed via `[tool.uv.sources]` pointing to `pytorch-cu128` index (compatible with CUDA 13.0 driver)
- **Demo model**: FinBERT (best macro-F1), loaded via `src/inference.load_best_model()` with fallback chain: FinBERT → BERT → DistilBERT
- **App style**: No emojis anywhere. Uses ASCII markers [+] positive, [~] neutral, [-] negative. Professional academic tone. User explicitly requested nothing that looks AI-generated.
- **Streamlit deps**: `lime`, `yfinance`, `streamlit-javascript`, `plotly` (already present). Removed `audio-recorder-streamlit` and `speechrecognition`.
- **Branches**: `main` = stable (Gradio app), `victoria` = Streamlit dashboard with new features

## Benchmark Results (2026-04-03)

| Rank | Model               | Accuracy | Macro-F1 | Type          | Train Time |
|------|---------------------|----------|----------|---------------|------------|
| 1    | FinBERT             | 0.8814   | 0.8766   | transformer   | 82s        |
| 2    | DistilBERT          | 0.8680   | 0.8600   | transformer   | 51s        |
| 3    | BERT                | 0.8546   | 0.8376   | transformer   | 111s       |
| 4    | TF-IDF + NaiveBayes | 0.7134   | 0.6268   | classical     | <1s        |
| 5    | TF-IDF + LogReg     | 0.7237   | 0.6009   | classical     | <1s        |
| 6    | BiLSTM              | 0.6371   | 0.3728   | deep_learning | 41s        |

**Key observations:**
- FinBERT's domain-specific pre-training gives it a clear edge over generic BERT
- DistilBERT beats full BERT (likely due to less overfitting on small dataset)
- Classical baselines are surprisingly competitive on accuracy but weak on minority classes (low macro-F1)
- BiLSTM struggles — small dataset + no pre-training = poor generalization, especially on negative class (0% recall)

## Commands

```bash
# Environment setup (finds uv automatically, even if not in PATH)
setup.bat

# Training (run.bat auto-discovers uv or .venv)
run.bat scripts/train_baselines.py           # CPU, ~10s
run.bat scripts/train_lstm.py                # CPU/GPU, ~40s
run.bat scripts/train_transformer.py --all   # GPU required, ~5 min total
run.bat scripts/train_transformer.py --model FinBERT  # Single model

# Leaderboard
run.bat scripts/build_leaderboard.py         # -> results/leaderboard.{csv,json,md}

# Demos
run.bat app.py                               # Gradio on http://localhost:7860
run.bat -m streamlit run streamlit_app.py    # Streamlit on http://localhost:8501

# Monitor GPU during training
nvidia-smi -l 5                              # Refresh every 5s

# Alternative: if uv IS in PATH, you can still use uv directly
uv sync
uv run python app.py
```

## Environment

- Python 3.11+
- uv for package management
- PyTorch 2.11.0+cu128 (CUDA 12.8, driver compat with CUDA 13.0)
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU (8 GB VRAM)
- OS: Windows 11

## What NOT to do

- Don't install PyTorch from default PyPI — it gives CPU-only. Use the `[tool.uv.sources]` config.
- Don't run transformer training without GPU — the script will exit with a clear error.
- Don't commit `results/models/` or `results/*_training/` — they're multi-GB.
- Don't change the label mapping without updating all scripts — it's centralized in `src/config.py`.
- Don't use absolute paths — everything is relative via `src/config.PROJECT_ROOT`.

## Self-Update Reminder

> **After finishing any task in this project, re-edit this CLAUDE.md** to reflect:
> 1. What changed (new files, modified behavior, new results)
> 2. Updated benchmark table if models were retrained
> 3. Any new decisions or gotchas discovered
>
> This ensures the next Claude Code session (or context refresh) has accurate, up-to-date project knowledge.
