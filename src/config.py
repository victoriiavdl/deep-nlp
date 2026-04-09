"""Central configuration for the project."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

DATA_CSV = DATA_DIR / "all-data.csv"

# ── Label scheme ───────────────────────────────────────────────────────
LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["positive", "neutral", "negative"]
NUM_CLASSES = len(LABEL_NAMES)

# ── Splitting ──────────────────────────────────────────────────────────
SEED = 42
TEST_SIZE = 0.2       # of total  → 80/20
VAL_RATIO = 0.125     # of train  → 80 * 0.125 = 10% of total → 70/10/20

# ── Transformer defaults ──────────────────────────────────────────────
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# ── LSTM defaults ──────────────────────────────────────────────────────
LSTM_MAX_VOCAB = 15_000
LSTM_MAX_SEQ_LEN = 50
LSTM_EMBED_DIM = 64
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 50

# ── TF-IDF defaults ──────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 20_000
TFIDF_NGRAM_RANGE = (1, 2)

# ── Model registry (for leaderboard) ─────────────────────────────────
TRANSFORMER_MODELS = {
    "DistilBERT": "distilbert-base-uncased",
    "BERT": "bert-base-uncased",
    "FinBERT": "ProsusAI/finbert",
}
