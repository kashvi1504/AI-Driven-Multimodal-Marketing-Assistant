"""
01_data_preprocessing.py
-------------------------
Reads every raw CSV from data/raw/, cleans the text, maps each dataset's
own label scheme onto one common scheme (negative / neutral / positive),
balances the classes, and writes a train/val/test split to
data/processed/.

Before running this, download the CSVs listed in README.md into
data/raw/ (exact filenames are in config.RAW_FILES). Any file that's
missing is simply skipped, with a message telling you what to add.

    python 01_data_preprocessing.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import LABEL2ID, PROCESSED_DIR, RAW_DIR, RAW_FILES, ROWS_PER_CLASS, SEED

np.random.seed(SEED)


# ── Step 1: text cleaning ────────────────────────────────────────────
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip the noise that doesn't carry sentiment: links, @handles,
    stray whitespace. Deliberately keeps hashtags, punctuation and
    emoji - those DO carry sentiment and the tokenizer handles them fine.
    """
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _raw_path(key: str) -> str:
    return os.path.join(RAW_DIR, RAW_FILES[key])


def _require_file(key: str) -> str:
    path = _raw_path(key)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"data/raw/{RAW_FILES[key]} not found. Download it (see README.md "
            f"for the link) and save it there with exactly that filename."
        )
    return path


# ── Step 2: per-dataset loaders (each returns a df with text,label) ──
def load_sentiment140() -> pd.DataFrame:
    """Kaggle: Sentiment140 - 1.6M tweets. Standard Kaggle export has NO
    header row, columns in this order: target,id,date,flag,user,text.
    target is 0=negative, 4=positive (no neutral class in this dataset)."""
    path = _require_file("sentiment140")
    # Some re-uploads of this dataset DO include a header - detect it by
    # checking whether the first cell of the first column is a number.
    first_row = pd.read_csv(path, encoding="latin-1", nrows=1, header=None)
    has_header = not str(first_row.iloc[0, 0]).strip().lstrip("-").isdigit()
    df = pd.read_csv(
        path,
        encoding="latin-1",
        header=0 if has_header else None,
        names=None if has_header else ["target", "id", "date", "flag", "user", "text"],
    )
    df.columns = [c.lower() for c in df.columns]
    target_col = "target" if "target" in df.columns else "sentiment"
    df["label"] = df[target_col].map({0: "negative", 4: "positive", 2: "neutral"})
    return df.dropna(subset=["label", "text"])[["text", "label"]]


def _map_emotion_to_sentiment(value: str):
    """The two Kaggle 'social/viral' datasets label posts with fine-grained
    emotions (Joy, Admiration, Frustration...) rather than plain
    negative/neutral/positive. This collapses them onto our 3 classes.
    Anything not recognised is dropped rather than guessed."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    positive_words = {
        "positive", "joy", "happiness", "happy", "excitement", "excited", "admiration",
        "gratitude", "contentment", "content", "love", "amusement", "enthusiasm",
        "hopeful", "hope", "pride", "elation", "satisfaction", "delight",
        "playful", "serenity", "empowerment", "acceptance", "affection",
        "compassion", "creativity", "inspired", "inspiration", "optimism",
        "thrill", "thrilled", "relief", "wonder", "kind", "kindness",
        "grateful", "cheerful", "proud", "confident", "confidence",
    }
    negative_words = {
        "negative", "sadness", "sad", "anger", "angry", "fear", "disgust",
        "frustration", "frustrated", "despair", "grief", "disappointment",
        "disappointed", "loneliness", "lonely", "bitter", "bitterness",
        "hate", "hatred", "jealousy", "resentment", "anxiety", "anxious",
        "regret", "bad", "embarrassed", "embarrassment", "melancholy",
        "shame", "guilt", "hurt", "heartbreak", "betrayal", "disgusted",
        "envy", "hopelessness", "insecurity", "isolation", "worry", "worried",
    }
    neutral_words = {
        "neutral", "confusion", "confused", "surprise", "surprised",
        "curiosity", "curious", "calm", "numbness", "numb", "nostalgia",
        "ambivalence", "indifference", "indifferent", "boredom", "bored",
        "contemplation", "reflective",
    }

    if v in positive_words:
        return "positive"
    if v in negative_words:
        return "negative"
    if v in neutral_words:
        return "neutral"
    return None


def _load_emotion_labeled_csv(key: str, text_cols, label_cols) -> pd.DataFrame:
    path = _require_file(key)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # these CSVs often ship with leading spaces in headers

    text_col = next((c for c in text_cols if c in df.columns), None)
    label_col = next((c for c in label_cols if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"Couldn't find expected columns in {path}. "
            f"Found columns: {list(df.columns)}. Update text_cols/label_cols "
            f"in 01_data_preprocessing.py to match."
        )

    df["label"] = df[label_col].astype(str).str.strip().apply(_map_emotion_to_sentiment)
    df = df.rename(columns={text_col: "text"}).dropna(subset=["label", "text"])
    return df[["text", "label"]]


def load_social_media_sentiments() -> pd.DataFrame:
    """Kaggle: Social Media Sentiments Analysis Dataset"""
    return _load_emotion_labeled_csv(
        "social_media_sentiments",
        text_cols=["Text", "text"],
        label_cols=["Sentiment", "sentiment"],
    )


def load_hf_sentiment() -> pd.DataFrame:
    """Hugging Face: syedkhalid076/Sentiment-Analysis, saved locally as CSV.
    Already labeled 0/1/2 = negative/neutral/positive."""
    path = _require_file("hf_sentiment")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    df["label"] = df["label"].map(id2label)
    return df.dropna(subset=["label", "text"])[["text", "label"]]


def generate_synthetic_marketing(n_per_class: int = 1500) -> pd.DataFrame:
    """Small templated generator for marketing-flavoured copy, so the
    model sees ad-style language and not just tweets. Fully local -
    nothing to download.

    NOTE on template/product counts: rows are deduplicated after
    generation (see combine_and_balance), so the number of USABLE rows
    per class is capped at len(templates) * len(products), not
    n_per_class. positive/negative/neutral all use the same template and
    product counts here so none of the three classes is starved relative
    to the others when no real datasets are downloaded."""
    products = [
        "this app", "our new skincare line", "the fitness tracker", "this course",
        "our coffee subscription", "the software update", "this ad campaign",
        "our loyalty program", "the mobile game", "this laptop",
        "the new headphones", "this meal kit", "our budgeting app", "the online class",
        "this backpack", "the smartwatch", "our email newsletter", "this camera",
        "the standing desk", "our cleaning service",
    ]
    positive_templates = [
        "I'm absolutely loving {p}, it's exceeded every expectation!",
        "{p} just made my day so much easier, highly recommend it.",
        "Can't believe how good {p} turned out, worth every penny.",
        "{p} is a game changer, five stars from me.",
        "Genuinely impressed with {p}, will be telling all my friends.",
        "{p} nailed it - exactly what I needed, no complaints at all.",
        "Switched to {p} last week and I'm already hooked.",
        "{p} is hands down the best purchase I've made this year.",
        "So glad I gave {p} a try, it's been fantastic so far.",
        "{p} works like a charm and the support team is great too.",
    ]
    neutral_templates = [
        "{p} is okay, does what it says on the box.",
        "Tried {p} today, nothing special but nothing bad either.",
        "{p} works fine, still deciding if I'll keep using it.",
        "Not sure yet how I feel about {p}, need more time with it.",
        "{p} arrived on time, haven't fully tested it yet.",
        "I used {p} this morning before heading out.",
        "Just picked up {p} from the store on my way home.",
        "{p} comes in a box about the size of a shoebox.",
        "I read the instructions for {p} before setting it up.",
        "{p} was released earlier this year.",
    ]
    negative_templates = [
        "Really disappointed with {p}, expected a lot more for the price.",
        "{p} broke after two days, asking for a refund.",
        "Would not recommend {p}, customer support was unhelpful too.",
        "{p} is way overpriced for what you actually get.",
        "Regret buying {p}, doesn't work as advertised at all.",
        "{p} stopped working within a week, total waste of money.",
        "Avoid {p} - it's nothing like what was promised online.",
        "{p} left me frustrated, the setup process was a nightmare.",
        "Not happy with {p} at all, already looking for alternatives.",
        "{p} feels cheaply made and the reviews definitely oversold it.",
    ]

    rows = []
    rng = np.random.default_rng(SEED)
    for label, templates in [
        ("positive", positive_templates),
        ("neutral", neutral_templates),
        ("negative", negative_templates),
    ]:
        for _ in range(n_per_class):
            template = rng.choice(templates)
            product = rng.choice(products)
            rows.append({"text": template.format(p=product), "label": label})
    return pd.DataFrame(rows)


def generate_factual_neutral(n: int = 6000) -> pd.DataFrame:
    """Plain factual / declarative statements with no opinion content at
    all - not tied to any product. This is the gap that makes models
    misclassify things like 'today is tuesday' as positive: sentiment
    datasets are almost entirely opinion text (reviews, tweets, ads), so
    genuinely neutral, fact-only sentences are underrepresented without
    this."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    cities = ["Mumbai", "Delhi", "Pune", "Bangalore", "Chennai", "Hyderabad", "Kolkata"]
    times = ["9 AM", "noon", "3 PM", "6 PM", "midnight", "10:30 AM", "2:15 PM"]
    foods = ["chocolate", "rice", "a sandwich", "pasta", "an apple", "tea", "coffee", "soup", "toast", "eggs"]
    places = ["the market", "the office", "the park", "the station", "college", "the gym", "the bank", "a friend's house"]
    media = ["a documentary", "the news", "a cricket match", "a podcast", "a movie", "a YouTube video"]
    activities = ["I had {food}.", "I went to {place}.", "I watched {media}.",
                  "I woke up at {time}.", "I walked to {place}.", "I took the bus to {place}.",
                  "I made {food} for breakfast.", "I finished my work at {time}.",
                  "I called a friend this afternoon.", "I read for a while before bed.",
                  "I charged my phone overnight.", "I printed the document at {time}.",
                  "I checked my email this morning.", "I packed my bag for {place}."]

    templates = [
        "Today is {day}.",
        "The meeting is scheduled for {day}.",
        "It is currently {time} here.",
        "The store opens at {time}.",
        "This event takes place in {month}.",
        "The report is due next {day}.",
        "The train departs from {city} at {time}.",
        "The office is located in {city}.",
        "The document has three pages.",
        "The package weighs two kilograms.",
        "The file was last updated in {month}.",
        "There are seven days in a week.",
        "The room has four chairs and a table.",
        "The bus route passes through {city}.",
        "The form needs to be submitted by {day}.",
        "The building has ten floors.",
        "The call is scheduled for {time}.",
        "The invoice number is printed at the top.",
        "The class starts at {time} on {day}.",
        "The parking lot is behind the building.",
    ] + activities

    rows = []
    rng = np.random.default_rng(SEED + 1)
    for _ in range(n):
        template = rng.choice(templates)
        text = template.format(
            day=rng.choice(days), month=rng.choice(months),
            city=rng.choice(cities), time=rng.choice(times),
            food=rng.choice(foods), place=rng.choice(places), media=rng.choice(media),
        )
        rows.append({"text": text, "label": "neutral"})
    return pd.DataFrame(rows)


# ── Step 3: combine, balance, split ──────────────────────────────────
def combine_and_balance(frames: list, rows_per_class: int, seed: int) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 2]
    df = df.drop_duplicates(subset=["text"])
    df = df[df["label"].isin(LABEL2ID.keys())]

    balanced = []
    for label in LABEL2ID.keys():
        subset = df[df["label"] == label]
        n = min(len(subset), rows_per_class)
        if len(subset) < rows_per_class:
            print(f"  ! only {len(subset)} rows available for '{label}' "
                  f"(wanted {rows_per_class}) - using all of them")
        balanced.append(subset.sample(n=n, random_state=seed))
    out = pd.concat(balanced, ignore_index=True)
    out["label_id"] = out["label"].map(LABEL2ID)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle


def main():
    print(f"Reading raw datasets from: {RAW_DIR}\n")

    loaders = [
        ("Sentiment140", load_sentiment140),
        ("Social Media Sentiments", load_social_media_sentiments),
        ("Sentiment Analysis (HF, saved as CSV)", load_hf_sentiment),
        ("Synthetic Marketing Sentiment (local, no file needed)", generate_synthetic_marketing),
        ("Factual Neutral Statements (local, no file needed)", generate_factual_neutral),
    ]

    frames = []
    for name, loader_fn in loaders:
        print(f"- {name}")
        try:
            df = loader_fn()
            print(f"    loaded {len(df):,} rows")
            frames.append(df)
        except Exception as e:
            print(f"    skipped ({e})")

    if not frames:
        sys.exit(
            "\nNo datasets loaded. Add at least one CSV to data/raw/ "
            "(see README.md for filenames + download links) and try again."
        )

    print("\nCombining, cleaning and balancing classes...")
    combined = combine_and_balance(frames, ROWS_PER_CLASS, SEED)
    print(combined["label"].value_counts())

    train_df, temp_df = train_test_split(
        combined, test_size=0.2, stratify=combined["label_id"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=SEED
    )

    for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        split_df.to_csv(path, index=False)
        print(f"Saved {name}: {len(split_df):,} rows -> {path}")

    print("\nDone. Next: python 02_model_training.py")


if __name__ == "__main__":
    main()