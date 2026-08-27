# Sentiment Studio — Marketing AI toolkit

A small pipeline that fine-tunes a real sentiment classifier on marketing
text, plus a Streamlit dashboard that wraps it together with heuristic
image-engagement scoring and a few image-editing tools. One folder for
raw data, one script per stage, one config file.

## What's implemented vs. planned

| Capability | Status | How |
|---|---|---|
| Text sentiment classification | **Implemented, trained model** | Fine-tuned DistilBERT/RoBERTa, evaluated on a held-out test split |
| Text engagement scoring | **Implemented, heuristic** | Rule-based (CTA/urgency keywords, length, punctuation) - not a trained model |
| Ad copy rewriting | **Implemented, hybrid** | Local flan-t5 generation when available, template fallback otherwise |
| Image engagement scoring | **Implemented, heuristic** | Pretrained CLIP zero-shot similarity + measurable visual features |
| Image style presets / auto-fix | **Implemented, deterministic** | Real pixel operations (brightness/contrast/color/sharpness/crop) |
| Generative image editing | **Implemented, real model** | Stable Diffusion img2img via `diffusers` |
| Precise image editing | **Implemented, real models** | `rembg` background removal, `easyocr` text detection |
| Canvas editor | **Implemented** | `streamlit-drawable-canvas` (crop/draw/text-box/flatten) |
| Pitt Ads dataset loading | **Implemented, loading only** | `07_pitt_ads_loader.py` - schema-agnostic loader, no engagement model built on it yet |
| Pitt Ads → Instagram engagement model | **Planned, not implemented** | See "Planned" section below - deliberately not built yet |

## File structure

```
config.py                    ← every path/setting lives here
01_data_preprocessing.py     ← loads, cleans, balances, splits sentiment data
02_model_training.py         ← fine-tunes the sentiment model + SentimentPredictor class
03_dashboard.py              ← Streamlit UI: sentiment, image analysis, editing tools
04_image_analysis.py         ← image engagement heuristic + deterministic style presets
05_generative_editing.py     ← Stable Diffusion img2img editing
06_image_editing.py          ← precise background swap (rembg) + text replace (easyocr)
07_pitt_ads_loader.py        ← Pitt Ads dataset loading interface (planned integration, see below)
Requirements.txt
data/
├── raw/                     ← put the downloaded CSVs here (see below)
└── processed/                ← train/val/test CSVs, auto-generated
models/
└── best_model/                ← saved after training
results/
├── classification_report.txt
├── confusion_matrix.png
└── checkpoints/
```

## 1. Install dependencies

```bash
pip install -r Requirements.txt
```

The last four lines of `Requirements.txt` (`rembg`, `easyocr`, `diffusers`,
`streamlit-drawable-canvas`) are **optional**. Everything else runs
without them - the dashboard detects what's missing and disables just
that page/feature with a clear message and an install hint, instead of
crashing. Check the sidebar's "System → Optional dependencies" panel to
see what's currently available.

## 2. Download the sentiment datasets (optional but recommended)

Download each one and save it into `data/raw/` with **exactly** the
filename shown — the script looks for these names.

| Dataset | Link | Save as |
|---|---|---|
| Sentiment140 | https://www.kaggle.com/datasets/kazanova/sentiment140 | `data/raw/sentiment140.csv` |
| Social Media Sentiments Analysis | https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset | `data/raw/social_media_sentiments.csv` |
| Sentiment Analysis Dataset | https://huggingface.co/datasets/syedkhalid076/Sentiment-Analysis (download the CSV from the "Files" tab) | `data/raw/hf_sentiment_analysis.csv` |

> **Note:** "Viral Social Media Trends & Engagement" is intentionally
> **not** in this list — it has no text/caption column and no sentiment
> label, only Views/Likes/Shares/Engagement_Level, so it can't feed the
> text classifier. See "Image engagement scoring" below for why it also
> isn't used for image engagement.

Kaggle downloads come as a `.zip` — unzip it, there's usually one CSV
inside; rename that CSV to the name above.

**You don't need any of these to get started.** Any file that's missing
is skipped automatically (with a message telling you what's missing) —
two more sources (synthetic marketing text + factual-neutral statements)
are generated in code either way, so `01_data_preprocessing.py` always
produces *something* to train on even with zero files downloaded. It
will be a small, template-based dataset in that case (a few hundred
rows per class) - fine for exercising the pipeline end-to-end, but add
the real datasets above for a classifier that generalizes well.

## 3. Run the pipeline

```bash
python 01_data_preprocessing.py     # cleans + balances + splits data
python 02_model_training.py         # fine-tunes, evaluates, saves best model
streamlit run 03_dashboard.py       # try it interactively
```

**You do not need to run 1-2 before trying the dashboard.** If
`models/best_model` doesn't exist yet, `03_dashboard.py` automatically
falls back to the public `cardiffnlp/twitter-roberta-base-sentiment-latest`
checkpoint (shown as a banner in the app), so you can explore the UI
immediately and swap in your own fine-tuned model later.

## Datasets used for sentiment

All sources get mapped onto one common 3-class scheme —
**negative / neutral / positive** — then balanced so no class dominates
training.

| Dataset | Notes |
|---|---|
| Sentiment140 | 1.6M tweets, binary (pos/neg) — no neutral examples in this one |
| Social Media Sentiments | Labels are fine-grained emotions (Joy, Frustration, etc.) — mapped onto our 3 classes automatically |
| Sentiment Analysis Dataset | Already 3-class (0/1/2) |
| Synthetic Marketing Sentiment | Generated locally, template-based — adds ad-style language |
| Factual Neutral Statements | Generated locally — plain declarative sentences with no opinion, fills a gap real sentiment corpora don't cover |

If a dataset's column names don't match what the script expects (Kaggle
occasionally tweaks these), it will print exactly which columns it found
so you can fix the `text_cols` / `label_cols` list in
`01_data_preprocessing.py` in one line.

## Model

Default: `distilbert-base-uncased` — laptop-CPU friendly, ~40% faster
than BERT/RoBERTa at similar accuracy. Change it with an environment
variable before training:

```bash
export SENTIMENT_MODEL_NAME="cardiffnlp/twitter-roberta-base-sentiment-latest"
python 02_model_training.py
```
Use the RoBERTa option when you have GPU access (college lab) — training
it on a CPU laptop will be very slow.

## Using the trained model elsewhere

`02_model_training.py` starts with a digit, so it can't be imported with
a plain `import` statement — that's expected, not a bug. Load it like this
from another script:

```python
import importlib.util
spec = importlib.util.spec_from_file_location("model_training", "02_model_training.py")
model_training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_training)

predictor = model_training.SentimentPredictor("models/best_model")
result = predictor.predict(["Boost your productivity with our AI tool!"])
# -> [{"label": "positive", "confidence": 0.94, "scores": {...}}]
```

If no fine-tuned model is found, `SentimentPredictor` automatically
falls back to the public `cardiffnlp/twitter-roberta-base-sentiment-latest`
checkpoint, so downstream modules never hard-crash on a missing model.

## Text engagement scoring — also a heuristic

The "Engagement" score shown alongside sentiment on the **Ad Copy** and
**Bulk Analysis** pages (`predict_engagement()` in `03_dashboard.py`) is
a **rule-based heuristic**, not a trained/validated model: it starts
from a baseline of 50 and adds or subtracts fixed amounts for CTA
keywords, urgency keywords, copy length, punctuation, and the
sentiment model's own class probabilities. It is labeled "Heuristic" in
the UI for the same reason the image score is — there is no real
labeled dataset of ad copy paired with actual engagement outcomes to
train or validate against. Treat "before/after" uplift numbers from the
rewrite suggestion the same way: directional, not a guaranteed result.

## Image engagement scoring — known limitation (worth citing in your report)

`04_image_analysis.py` and the **Image Analysis** dashboard page score
images using CLIP semantic similarity + measurable visual features
(brightness, contrast, saturation, sharpness, aspect ratio). This is a
**heuristic**, not a trained/validated model — there was no dataset
available pairing real marketing images with real engagement outcomes
at a usable scale.

We investigated using the "Viral Social Media Trends & Engagement" Kaggle
dataset (Post metadata + Views/Likes/Shares/Comments/Engagement_Level) as
a real source to predict engagement numbers from. Statistical analysis
showed this dataset is very likely randomly generated rather than real
data:

- `Views` standard deviation (1,459,490) almost exactly matches the
  theoretical standard deviation of a uniform random distribution over
  its range (1,442,846) — the signature of synthetic data, not real
  engagement (which is typically power-law distributed with outliers).
- Mean views are nearly identical across all 4 platforms (2.40M-2.55M),
  which real platforms would not produce.
- Mean views by `Engagement_Level` are inverted from what the label
  implies (`High` = 2.45M, `Low` = 2.51M) — proving no real relationship
  between the label and the underlying metrics.

Given this, we chose **not** to build a model or present engagement-count
predictions (likes/comments/shares) from this data, since doing so would
produce confident-looking but meaningless numbers. The dashboard shows a
0-100 heuristic score and High/Medium/Low tier only, clearly labeled as
non-validated, rather than fabricated counts. This is a deliberate,
documented methodological choice, not an oversight.

## Optional dependencies and how they degrade

| Feature | Package | If missing |
|---|---|---|
| Ad copy rewriting (LLM-generated) | `transformers` + `sentencepiece` (flan-t5) | Falls back to a deterministic template rewriter — never crashes |
| Generative Editing page | `diffusers`, `accelerate` | Page shows an install hint instead of the editor |
| Precise Editing → background swap | `rembg` | That action shows a friendly error with an install hint |
| Precise Editing → text detection | `easyocr` | That action shows a friendly error with an install hint |
| Canvas Editor page | `streamlit-drawable-canvas` | Page shows an install hint instead of the canvas |

## Planned: Pitt Ads + Instagram engagement integration

We're extending the image-analysis component with two additional
datasets, but this integration is **not implemented yet** - only the
loading interface for one of them is.

**Pitt Ads** (`creative-graphic-design/PittImageVideoAdsDataset` on
Hugging Face) provides ~65k image ads with crowdsourced annotations:
topics, sentiments, persuasive strategies, symbolic references, slogans,
and action/reason Q&A. `07_pitt_ads_loader.py` implements a schema-agnostic
loading interface for it (`load_pitt_ads()`, `describe_schema()`) — it
does not hardcode exact column names, since dataset cards can change
between `datasets` library versions; call `describe_schema()` to see
what's actually there before writing code against specific fields.

**Instagram influencer dataset** (held separately, not included in this
repo) has real post images with engagement metadata (likes, comments,
follower counts).

**Why these are not combined into one engagement model yet:** Pitt Ads
describes advertising *content/semantics* (what the ad is about, its
persuasive strategy) with no engagement outcomes. The Instagram dataset
has real engagement outcomes with no ad-strategy annotations. They are
two different images, from two different distributions, describing two
different things - there's no evidence-based way to map one dataset's
labels onto the other's outcomes. Building a fabricated Pitt→Instagram
mapping (e.g., assuming "positive sentiment ads get more likes") would
produce a confident-looking but scientifically unsupported model, which
is exactly the mistake this project avoided with the Viral Content
dataset above.

The planned, honest path forward:
1. Extract the **same kind of visual/content features** from both
   datasets independently (CLIP embeddings, visual features, and - once
   `describe_schema()` confirms them - relevant Pitt Ads annotations).
2. Train an engagement model **only on the Instagram dataset**, since
   that's the only one with real engagement labels, using Pitt-Ads-style
   features as additional (not assumed-equivalent) inputs where they
   transfer.
3. Report validation metrics honestly (accuracy/R²/whatever is
   appropriate) on that trained model — only once it exists — instead
   of the current heuristic score.
4. Keep this as an additive page in the dashboard once it exists,
   clearly distinguished from the current heuristic ("Predicted" vs
   "Heuristic" labeling, matching the convention already used
   throughout the UI).

None of this is implemented in the current codebase - do not assume any
Pitt Ads or Instagram data currently influences the dashboard's
engagement scores.

## Dashboard pages

**Analyze**
- **Ad Copy** — sentiment + heuristic engagement + explanation + rewrite suggestion for one piece of copy.
- **Image Analysis** — heuristic engagement score for an uploaded image, visual-feature breakdown, and deterministic style presets (auto-fix, warm tone, punchy, etc.) with before/after comparison.
- **Bulk Analysis** — same sentiment/engagement pipeline over a CSV of copy.

**Create & Edit**
- **Generative Editing** — Stable Diffusion img2img restyling from a text prompt.
- **Precise Editing** — exact background color swap (`rembg`) and OCR-based text replacement (`easyocr`), for when you need an exact result rather than a generative approximation.
- **Canvas Editor** — freehand draw, crop, and text-box tools on an uploaded image.

The sidebar also shows which sentiment model is active and which
optional dependencies are currently installed, so you always know what
you're looking at before trusting a number.

## Running tests/checks

```bash
python -m py_compile *.py             # syntax check
python 01_data_preprocessing.py       # should always succeed - see step 2
python 02_model_training.py           # needs data/processed/*.csv from step above
streamlit run 03_dashboard.py --server.fileWatcherType none
```
