# Financial News Sentiment Analysis

Benchmarking classical ML, deep learning, and transformer models on the [FinancialPhraseBank](https://www.researchgate.net/publication/251231364_FinancialPhraseBank) dataset.

## Présentation

**Financial Sentiment Analyzer** est un pipeline NLP complet qui compare **6 modèles** sur 3 paradigmes pour la classification de sentiment de titres d'actualité financière.

L'application permet de :
- **Entraîner et évaluer** des modèles classiques (TF-IDF), deep learning (BiLSTM) et transformers (DistilBERT, BERT, FinBERT)
- **Visualiser** les performances via un leaderboard interactif
- **Prédire** le sentiment en temps réel avec explainabilité LIME
- **Analyser** en batch des fichiers CSV ou le flux d'actualités Yahoo Finance

## Données utilisées

| Source | Description |
|--------|-------------|
| **FinancialPhraseBank** | 4 846 titres financiers annotés par des experts (3 classes : positive, neutre, négative) |
| **Yahoo Finance** | Flux d'actualités en temps réel (via yfinance) |

Le dataset est déséquilibré (neutre 59.4%, positive 28.1%, négative 12.5%), le **macro-F1** est donc la métrique principale.

## Résultats

| Rang | Modèle | Accuracy | Macro-F1 | Type |
|------|--------|----------|----------|------|
| 1 | FinBERT | 88.1% | 87.7% | Transformer |
| 2 | DistilBERT | 86.8% | 86.0% | Transformer |
| 3 | BERT | 85.5% | 83.8% | Transformer |
| 4 | TF-IDF + NaiveBayes | 71.3% | 62.7% | Classical ML |
| 5 | TF-IDF + LogReg | 72.4% | 60.1% | Classical ML |
| 6 | BiLSTM | 63.7% | 37.3% | Deep Learning |

FinBERT, pré-entraîné sur du texte financier, surpasse les modèles généralistes, illustrant l'apport de l'adaptation au domaine.

## Structure du projet

```
├── data/                      # Dataset FinancialPhraseBank
├── src/                       # Modules Python (config, data, metrics, inference)
├── scripts/                   # Scripts d'entraînement et leaderboard
│   ├── train_baselines.py     # TF-IDF + Logistic Regression / Naive Bayes
│   ├── train_lstm.py          # BiLSTM (TensorFlow)
│   ├── train_transformer.py   # DistilBERT / BERT / FinBERT (PyTorch)
│   └── build_leaderboard.py   # Génération du leaderboard
├── streamlit_app.py           # Dashboard interactif
├── presentation/              # Slides Beamer (.tex + .pdf)
├── rapport/                   # Rapport écrit (.tex + .pdf)
├── results/                   # Résultats (JSON, CSV, Markdown)
└── pyproject.toml             # Dépendances & métadonnées
```

## Stack technique

- **Python 3.11+**
- **PyTorch** — fine-tuning des transformers (fp16, gradient checkpointing)
- **TensorFlow** — entraînement BiLSTM
- **Hugging Face Transformers** — DistilBERT, BERT, FinBERT
- **scikit-learn** — baselines classiques (TF-IDF)
- **Streamlit** — dashboard interactif
- **LIME / Plotly** — explainabilité et visualisations

## Installation locale

```bash
git clone https://github.com/victoriiavdl/deep-nlp.git
cd deep-nlp

# Avec uv (recommandé)
uv sync
uv run python scripts/train_baselines.py
uv run python scripts/train_lstm.py
uv run python scripts/train_transformer.py --all
uv run python scripts/build_leaderboard.py

# Lancer le dashboard
uv run streamlit run streamlit_app.py

# Ou avec pip
python -m venv .venv && .venv/Scripts/activate
pip install -e .
streamlit run streamlit_app.py
```

> **Note :** GPU (CUDA) requis pour l'entraînement des transformers. CPU suffisant pour les baselines et BiLSTM.

## Contexte

Ce projet a été réalisé dans le cadre d'un cours de **Deep Learning / NLP** (Master MoSEF). L'objectif était de benchmarker l'évolution des techniques NLP — des approches bag-of-words aux transformers pré-entraînés — sur une tâche de classification de sentiment en domaine financier.
