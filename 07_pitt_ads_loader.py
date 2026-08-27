"""
07_pitt_ads_loader.py
------------------------
Loading interface for the Pitt Image Ads dataset
(`creative-graphic-design/PittImageVideoAdsDataset` on Hugging Face).

STATUS: loading/inspection only. This file does NOT predict engagement
and does NOT connect Pitt Ads labels to the Instagram engagement
dataset. See "Planned: Pitt Ads + Instagram integration" in README.md
for why that mapping is deliberately not implemented yet.

Pitt Ads ("Automatic Understanding of Image and Video Advertisements",
Hussain et al.) provides ~64k image ads with crowdsourced annotations:
topics, sentiments, persuasive strategies, symbolic references, slogans,
and action/reason Q/A. It has NO engagement metrics (likes/comments/CTR) -
it is a content/semantics dataset, not an outcomes dataset. The separate
Instagram influencer dataset has real engagement metadata but no ad-style
annotations. The two datasets describe different things about different
images; treat them as two independent feature sources, not as a matched
pair.

This module deliberately does NOT hardcode the exact column names the
Hugging Face dataset exposes. Dataset cards drift over time, and
guessing wrong would silently produce empty/broken columns instead of a
clear error. Call `describe_schema()` first to see what's actually
there on your installed `datasets` version before writing code that
depends on specific field names.

Usage:
    from importlib import import_module
    pitt = import_module("07_pitt_ads_loader")

    pitt.describe_schema()                  # prints available columns
    df = pitt.load_pitt_ads(limit=200)       # pandas DataFrame, first 200 rows
"""

import os

HF_DATASET_ID = "creative-graphic-design/PittImageVideoAdsDataset"

# The dataset ships two configs on Hugging Face: "image_ads" and
# "video_ads", each with a single "train" split (no val/test split is
# provided upstream).
DEFAULT_CONFIG = os.environ.get("PITT_ADS_CONFIG", "image_ads")
DEFAULT_SPLIT = "train"


def _load_raw(config_name: str = DEFAULT_CONFIG, split: str = DEFAULT_SPLIT, streaming: bool = True):
    """Returns a raw HF `datasets` Dataset/IterableDataset, or raises a
    clear error if `datasets` isn't installed or the download fails
    (no network, dataset renamed/removed, etc). Streaming is on by
    default - this is a ~65k row, multi-gigabyte (images included)
    dataset, so callers should not eagerly materialize all of it."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The `datasets` package is required to load Pitt Ads. "
            "Install it with: pip install datasets"
        ) from e

    try:
        return load_dataset(HF_DATASET_ID, name=config_name, split=split, streaming=streaming)
    except Exception as e:
        raise RuntimeError(
            f"Couldn't load '{HF_DATASET_ID}' (config='{config_name}', split='{split}') "
            f"from Hugging Face. This usually means no internet access, the dataset "
            f"was renamed/removed, or the installed `datasets` version doesn't support "
            f"the format it's stored in. Original error: {e}"
        ) from e


def describe_schema(config_name: str = DEFAULT_CONFIG, n: int = 1) -> dict:
    """Fetches `n` example(s) (streaming, so this does not download the
    whole dataset) and returns {column_name: python_type_name}. Print
    this before building anything that depends on specific field names -
    it is the source of truth, not this file's docstring."""
    raw = _load_raw(config_name=config_name, streaming=True)
    it = iter(raw)
    sample = [next(it) for _ in range(n)]
    schema = {k: type(v).__name__ for k, v in sample[0].items()}
    for key, value in schema.items():
        print(f"  {key:30s} {value}")
    return schema


def load_pitt_ads(config_name: str = DEFAULT_CONFIG, split: str = DEFAULT_SPLIT, limit: "int | None" = 200):
    """Materializes up to `limit` rows into a pandas DataFrame, keeping
    whatever columns the dataset actually has (no renaming/reshaping -
    that belongs in dataset-specific preprocessing downstream, kept
    separate from this loader on purpose). Pass limit=None to attempt
    loading everything (slow, large - only do this with real disk/time
    budget for it).
    """
    import pandas as pd

    streaming = limit is not None
    raw = _load_raw(config_name=config_name, split=split, streaming=streaming)

    if streaming:
        rows = []
        for i, row in enumerate(raw):
            if i >= limit:
                break
            rows.append(row)
        return pd.DataFrame(rows)

    return raw.to_pandas()


if __name__ == "__main__":
    print(f"Inspecting schema for config='{DEFAULT_CONFIG}', split='{DEFAULT_SPLIT}'...")
    try:
        describe_schema()
    except (ImportError, RuntimeError) as e:
        print(f"Could not inspect schema: {e}")
