"""
config.py
---------
One place for every path, label, and setting the pipeline uses.
Change things here instead of hunting through 3 different scripts.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────
# Everything is relative to this file, so the project runs the same
# no matter which machine (laptop / college PC) you clone it onto.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "best_model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [RAW_DIR, PROCESSED_DIR, os.path.dirname(MODEL_DIR), RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Labels ─────────────────────────────────────────────────────────
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3

# ── Raw dataset files ────────────────────────────────────────────────
# Just drop these files into data/raw/ with these exact names.
# See README.md for the download link for each one.
RAW_FILES = {
    "sentiment140": "sentiment140.csv",
    "social_media_sentiments": "social_media_sentiments.csv",
    "hf_sentiment": "hf_sentiment_analysis.csv",
    # no file needed for "synthetic" - it's generated in code
    # "viral_content" was intentionally dropped - see README
    # ("Image engagement scoring" section) for why.
}

# How many rows to keep PER CLASS after combining everything.
# Keeps training time reasonable on a laptop CPU. Raise this when
# running on a college GPU machine.
ROWS_PER_CLASS = int(os.environ.get("SENTIMENT_ROWS_PER_CLASS", 8000))

# ── Model ──────────────────────────────────────────────────────────
# distilbert is ~40% faster than BERT/RoBERTa and fine on a laptop CPU.
# Swap to "cardiffnlp/twitter-roberta-base-sentiment-latest" when you
# have GPU access (college lab) for a real accuracy boost.
MODEL_NAME = os.environ.get("SENTIMENT_MODEL_NAME", "distilbert-base-uncased")

MAX_LENGTH = 128
SEED = 42