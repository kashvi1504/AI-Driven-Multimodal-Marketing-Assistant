# Sentiment Studio — Marketing AI toolkit

A small pipeline that fine-tunes a real sentiment classifier on marketing
text, plus a local web app (FastAPI backend + a hand-built HTML/CSS/JS
front end) that wraps it together with creative-quality scoring for ad
images and a few image-editing tools. One folder for
raw data, one script per stage, one config file.

## What's implemented vs. planned

| Capability | Status | How |
|---|---|---|
| Text sentiment classification | **Implemented, trained model** | Fine-tuned DistilBERT/RoBERTa, evaluated on a held-out test split |
| Text engagement scoring | **Implemented, heuristic** | Rule-based (CTA/urgency keywords, length, punctuation) - not a trained model |
| Ad copy rewriting | **Implemented, hybrid + grounded** | Local flan-t5 generation when available, template fallback otherwise; every rewrite passes a grounding check so it can't invent offers, prices or deadlines |
| Image quality scoring | **Implemented, heuristic** | Measured focus/exposure/contrast/noise/resolution/framing with critical-defect caps, + CLIP zero-shot appeal (see "Image quality scoring") |
| Image style presets / auto-fix | **Implemented, deterministic** | Real pixel operations (brightness/contrast/color/sharpness/crop) |
| Generative image editing | **Implemented, real model** | Stable Diffusion img2img via `diffusers` |
| Precise image editing | **Implemented, real models** | `rembg` background removal, `easyocr` text detection |
| Canvas editor | **Implemented** | In-browser HTML canvas (draw/line/box/circle/text box/crop, undo, PNG export) |
| Pitt Ads dataset loading | **Implemented, loading only** | `07_pitt_ads_loader.py` - schema-agnostic loader, no engagement model built on it yet |
| Instagram image engagement model | **Implemented, trained (needs your data)** | `10_image_model_training.py` - CLIP embeddings + logistic regression, account-grouped CV against honest baselines. Shown on the Image Analysis page once `models/image_engagement.joblib` exists |
| Pitt Ads → Instagram engagement model | **Planned, not implemented** | See "Planned" section below - deliberately not built yet |
| Instagram dataset authenticity audit | **Implemented, statistical checks** | `08_instagram_audit.py` - distribution, Benford, consistency and duplicate checks before any modelling |
| Website SEO/GEO analysis | **Implemented, heuristic checklist** | `09_website_analysis.py` + "Website Audit" web page; Google PageSpeed Insights for performance |

## File structure

```
config.py                    ← every path/setting lives here
01_data_preprocessing.py     ← loads, cleans, balances, splits sentiment data
02_model_training.py         ← fine-tunes the sentiment model + SentimentPredictor class
03_web_app.py                ← web app: FastAPI backend, run it and open http://127.0.0.1:8000
copy_analysis.py             ← ad-copy features, engagement heuristic, grounded rewriter (used by the web app)
04_image_analysis.py         ← image quality checks (blur, exposure, ...) + deterministic style presets
05_generative_editing.py     ← Stable Diffusion img2img editing
06_image_editing.py          ← precise background swap (rembg) + text replace (easyocr)
07_pitt_ads_loader.py        ← Pitt Ads dataset loading interface (planned integration, see below)
08_instagram_audit.py        ← authenticity audit for the Instagram dataset (run before modelling)
09_website_analysis.py       ← single-page SEO/GEO/performance audit (CLI + web page)
10_image_model_training.py   ← trains + validates an image engagement model on the Instagram dataset
web/                         ← the site: index.html, styles.css, app.js (no build step)
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

The last three lines of `Requirements.txt` (`rembg`, `easyocr`,
`diffusers`) are **optional**. Everything else runs without them - the
web app detects what's missing and disables just that feature with a
clear message and an install hint, instead of crashing. Check the
sidebar's "System → Optional features" panel to see what's available.

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
python 03_web_app.py                # then open http://127.0.0.1:8000
```

The web app runs entirely on your machine (`--port 8080` to change the
port). The sentiment model and CLIP warm up in the background when it
starts; heavier models (Stable Diffusion, OCR, the rewriter) load the
first time you use them. API docs are at http://127.0.0.1:8000/api/docs.

**You do not need to run 1-2 before trying the web app.** If
`models/best_model` doesn't exist yet, `03_web_app.py` automatically
falls back to the public `cardiffnlp/twitter-roberta-base-sentiment-latest`
checkpoint (shown in the sidebar), so you can explore the UI
immediately and swap in your own fine-tuned model later.

## Datasets used for sentiment

All sources get mapped onto one common 3-class scheme —
**negative / neutral / positive** — then balanced so no class dominates
training.

| Dataset | `source` key | How labels were made (`label_method`) | Synthetic text? | Used for |
|---|---|---|---|---|
| Sentiment140 | `sentiment140` | `emoticon` — assigned automatically from emoticons in the tweet (distant supervision), **not** human-annotated; no neutral class | No | train + val/test |
| Social Media Sentiments | `social_media_sentiments` | `emotion_mapped` — fine-grained emotions (Joy, Frustration…) mapped by us onto 3 classes | No | train + val/test |
| Sentiment Analysis Dataset | `hf_sentiment` | `provided_3class` — dataset ships its own 0/1/2 labels | No | train + val/test |
| Synthetic Marketing Sentiment | `synthetic_marketing` | `template` — text and label both generated by our templates | **Yes** | **train only** |
| Factual Neutral Statements | `factual_neutral` | `template` — plain declarative sentences, generated locally | **Yes** | **train only** |

### Structured dataset schema

Every row written to `data/processed/*.csv` has the same columns
(defined once in `config.SCHEMA_COLUMNS` and checked by
`validate_schema()` in `01_data_preprocessing.py` — the script stops
with an error if any split breaks the schema):

| Column | Meaning |
|---|---|
| `text` | cleaned text |
| `label` / `label_id` | negative / neutral / positive and 0 / 1 / 2 |
| `source` | which dataset the row came from (table above) |
| `label_method` | how the label was produced (table above) |
| `is_synthetic` | `True` if the text was generated by our own templates |
| `original_label` | the label exactly as it appeared in the source (e.g. `4`, `Joy`, `template:positive`) |

Provenance for each source lives in `config.SOURCE_INFO`, so adding a
new dataset means adding one entry there plus one loader function.

### Evaluation uses real text only

Synthetic template rows are **never** placed in the validation or test
split. Before this change they were split randomly, so the same 10
templates × 20 products appeared in both train and test, and test
accuracy partly measured template memorisation. Now:

- validation + test (20% of the data, 50/50) are drawn only from the
  real sources in `config.EVAL_SOURCES`;
- all synthetic rows go to training;
- `data/processed/dataset_summary.csv` records rows per split × source ×
  label, and `split_info.json` records whether evaluation is real-only.

If **no** real dataset has been downloaded, there is nothing real to
evaluate on. The script then falls back to a random split and prints a
loud warning, and the training report is headed "metrics are inflated —
do not report them".

### Per-source metrics

`02_model_training.py` writes:

- `results/classification_report.txt` — a header stating what the test
  set contains, the overall report, then the per-source breakdown;
- `results/per_source_metrics.csv` — accuracy, weighted-F1 and macro-F1
  for each source, for "ALL real" and "ALL synthetic", and overall, with
  per-class row counts.

The best checkpoint is now chosen on **macro-F1**, so each class counts
equally, rather than on weighted F1.

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

## Ad copy rewriting — grounding check (anti-hallucination)

The rewriter must not make claims the advertiser never made. Earlier
versions told the model to add urgency ("today only", "limited time")
and then force-appended "today only", "Try now!" and "Feel the
difference!" to every candidate. That invented sales and deadlines
(false advertising), and it also inflated the heuristic score, which
rewards exactly those words. Now:

- The prompt forbids adding prices, numbers, discounts, deadlines,
  guarantees or claims, and says there is no offer unless one is given.
- The **Ad Copy** page has an optional **"Real offer details"** box. Only
  offers typed there, or already present in the copy, may appear in a
  rewrite. In **Bulk Analysis**, an optional `offer` / `offer_details`
  CSV column does the same per row.
- Every generated candidate goes through `find_unsupported_claims()`.
  Any candidate containing a number, price, percentage, date or deadline
  word, or a promotional claim (`sale`, `% off`, `free`, `guaranteed`,
  `#1`, `proven`…) that is not in the user's input is **rejected**. The
  UI lists the rejected rewrites and the claims that were invented.
- The deterministic fallback (`safe_template_rewrite`) only restructures
  the user's own words and adds a neutral call-to-action ("Learn more").
- "Uplift" is renamed to **"heuristic score change"**. It comes from the
  same keyword heuristic, not a measured engagement result.

## Text engagement scoring — also a heuristic

The "Engagement" score shown alongside sentiment on the **Ad Copy** and
**Bulk Analysis** pages (`predict_engagement()` in `copy_analysis.py`) is
a **rule-based heuristic**, not a trained/validated model: it starts
from a baseline of 50 and adds or subtracts fixed amounts for CTA
keywords, urgency keywords, copy length, punctuation, and the
sentiment model's own class probabilities. It is labeled "Heuristic" in
the UI for the same reason the image score is — there is no real
labeled dataset of ad copy paired with actual engagement outcomes to
train or validate against. Treat "before/after" uplift numbers from the
rewrite suggestion the same way: directional, not a guaranteed result.

## Image quality scoring

`04_image_analysis.py` and the **Image Analysis** page rate whether a
creative is *technically fit to run as an ad*. It is a **heuristic**, not
a model trained on engagement outcomes (that's `10_image_model_training.py`).

**What's measured** (at feed display size, long side ≤ 1080 px - what a
viewer actually sees):

| Check | Method | Weight | Caps the score? |
|---|---|---|---|
| Focus | No-reference re-blur metric (Crété-Roffet et al., 2007) on the sharpest ~10% of textured regions, so a sharp product on a deliberately soft background passes | 35% | Yes - at 30 ("low") |
| Exposure | 1st/99th luminance percentiles: only images with no highlights (underexposed) or no shadows (washed out) fail, so dark luxury or white-background ads aren't penalised for their style | 20% | Yes - at 30 |
| Resolution | Shortest side (fails below ~200 px, passes from 600 px) | 15% | Yes - at 30 |
| Contrast | Tonal range (1st-99th percentile) | 15% | Yes - at 45 |
| Noise | Immerkær fast noise estimate, ignoring clipped pure black/white areas | 10% | Yes - at 45 |
| Framing | Distance to a standard ad ratio (1:1, 4:5, 3:4, 2:3, 9:16, 4:3, 16:9, 1.91:1) | 5% | No |

Colour saturation is reported but **not scored** - muted, monochrome
and black-and-white ads are a style choice, not a defect.

The technical score (weighted mean) is blended 70/30 with a CLIP
zero-shot "polished ad vs poor image" appeal score (prompts are worded
to be neutral about photo vs graphic design). Then **critical defects
cap the result**: a badly blurred, badly exposed or tiny image is always
rated "low", however good everything else is. Earlier versions averaged
sharpness with four other features, so even very blurry photos landed
on a neutral "medium" score.

Thresholds were calibrated on real photos and graphic ads at controlled
blur (Gaussian σ 0-10), noise, exposure, contrast and resolution levels:
every image blurred at σ ≥ 2 rates "low"; sharp originals, flat-design
ads, dark-themed ads and white-background product shots rate "high".
Known limit: grain is judged at feed size, so fine grain on a very large
image (which downscaling smooths out) isn't penalised.

### Why there's no engagement *prediction* from images by default

There was no dataset available pairing real marketing images with real
engagement outcomes at a usable scale.

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
produce confident-looking but meaningless numbers. The web app shows a
0-100 heuristic score and Good/Fair/Needs work tier only, clearly labeled as
non-validated, rather than fabricated counts. This is a deliberate,
documented methodological choice, not an oversight.

## Optional dependencies and how they degrade

| Feature | Package | If missing |
|---|---|---|
| Ad copy rewriting (LLM-generated) | `transformers` + `sentencepiece` (flan-t5) | Falls back to a deterministic template rewriter — never crashes |
| Generative Editing page | `diffusers`, `accelerate` | Page shows an install hint instead of the editor |
| Precise Editing → background swap | `rembg` | That action shows a friendly error with an install hint |
| Precise Editing → text detection | `easyocr` | That action shows a friendly error with an install hint |

## Website SEO/GEO analysis

`09_website_analysis.py` audits **one web page** for three things:
**SEO** (search engine optimisation), **GEO** (generative engine
optimisation, i.e. how easy the page is for AI answer engines such as
ChatGPT, Perplexity and Google AI Overviews to read, trust and cite) and
**performance**. It also runs the project's sentiment model over the
page copy. Use it from the **Website Audit** page of the web app, or from
the command line:

```bash
python 09_website_analysis.py https://example.com
python 09_website_analysis.py example.com --no-pagespeed --no-sentiment
python 09_website_analysis.py https://example.com --json results/example.json
```

### SEO checks

| Check | Pass | Warn | Fail |
|---|---|---|---|
| Title tag | 30–60 characters | present, other length | missing |
| Meta description | 70–160 characters | present, other length | missing |
| Single H1 | exactly one | more than one | none |
| Heading order | starts at h1, no skipped levels | 1–2 skips (e.g. h2→h4) | 3+ skips / no headings |
| Image alt text | ≥ 90% of images have `alt` | 50–89% | < 50% |
| Canonical tag | `rel=canonical` present | missing | – |
| Mobile viewport | viewport meta present | – | missing |
| HTTPS | final URL is https | – | http |
| Broken links | 0 broken in a sample of up to 20 | 1–2 broken | 3+ broken |
| Word count (main content) | ≥ 300 words | 100–299 | < 100 |

Links answering 401/403/429 are reported as "refused automated checks",
not as broken, because many sites block bots rather than being broken.

### GEO checks

The GEO checks are based on Aggarwal et al., *"GEO: Generative Engine
Optimization"* (KDD 2024, arXiv:2311.09735). The paper found that adding
**citations, quotations and statistics** were among the content changes
that most improved a source's visibility in generative-engine answers.

| Check | What it looks for |
|---|---|
| Structured data | schema.org JSON-LD blocks, and which types (Organization, Article, FAQPage…) |
| FAQ / question headings | question-style headings (`How…?`, `What…?`) or FAQPage schema |
| Statistics & numbers | percentages, amounts and figures in the main copy |
| Outbound citations | links to other domains (sources the page cites) |
| Quotations | `<blockquote>`/`<q>` or quoted sentences |
| Author & date | author and published/updated date in metadata, JSON-LD or `<time>` |
| AI crawler access | whether `robots.txt` blocks GPTBot, ClaudeBot, PerplexityBot or Google-Extended |

### Performance

Performance comes from Google's free **PageSpeed Insights** API: the
Lighthouse performance score (0–100) plus Core Web Vitals — LCP
(loading), CLS (visual stability) and INP (responsiveness). The module
uses real-user "field" data when Google has it for the site, otherwise
Lighthouse "lab" data, and labels which one was used. INP only exists as
field data, so small sites often show "no data" for it. The API works
without a key but is rate-limited; set `PAGESPEED_API_KEY` (read in
`config.py`) for reliable use. If the call fails, the section says
"unavailable" and the rest of the audit still runs.

### Scoring — what "heuristic" means here

The SEO and GEO scores are **weighted checklist scores**. Each check
earns pass = 1, warn = 0.5, fail = 0 points, multiplied by its weight.
Score = points ÷ weights of the checks that ran × 100. The weights live
in `SEO_WEIGHTS` / `GEO_WEIGHTS` at the top of `09_website_analysis.py`
and are a documented judgement call, not fitted to data. **They do not
predict search rankings or AI citations** — no ranking or citation data
is used anywhere. Only the performance score is Google's own
measurement.

### Limits

- **Single page only** — not a site crawl; other pages may differ.
- **No ranking, traffic or citation data** — it checks the page, not how
  it performs in search or AI answers.
- **JavaScript-rendered content may be missed** — the page is fetched as
  raw HTML, so content that only appears after JavaScript runs isn't seen.
- Sites that block bots (HTTP 403/429) can't be audited.
- Broken-link checking samples at most 20 links.

## Image engagement model (Instagram)

`10_image_model_training.py` trains a real image engagement model on the
Instagram influencer dataset - the only source with real engagement
outcomes. Run the authenticity audit first:

```bash
python 08_instagram_audit.py
python 10_image_model_training.py                    # image paths in the CSV
python 10_image_model_training.py --download-images  # image URLs in the CSV
```

Put the CSV at `data/raw/instagram_influencer.csv` (or set
`INSTAGRAM_RAW_FILE`) and the image files in `data/raw/instagram_images/`
(or set `INSTAGRAM_IMAGE_DIR`). Columns are auto-detected; override with
`--account-col`, `--image-col`, `--likes-col`, `--followers-col`.

**What it predicts:** whether an image would land in its creator's top or
bottom third of posts by engagement rate (likes + comments ÷ followers).
Ranking *within* an account removes follower count, niche and posting
habits, so what's left to learn from is the image. It does not predict
absolute likes. Posts from the dataset's last 7 days are dropped (still
collecting likes) and so are accounts with fewer than 6 usable posts.

**Model:** frozen CLIP (`openai/clip-vit-base-patch32`, same as the
heuristic) image embeddings + the visual features from
`04_image_analysis.py` → standardised logistic regression. Embeddings are
cached in `data/processed/clip_image_embeddings.npz`, so re-runs are fast.

**Evaluation:** 5-fold `GroupKFold` by account - test accounts are never
seen in training. Out-of-fold AUC and within-account Spearman are reported
with 95% CIs from resampling accounts, next to these baselines:

| Baseline | Why it's there |
|---|---|
| Current heuristic (`score_image`) | the model has to beat what the web app shows today |
| Visual features only | shows how much CLIP adds |
| Permuted labels | should be ~0.5 AUC - if not, something leaks |
| Random (ungrouped) split | reference only: shows how much a leaky split would flatter the model |

Results go to `results/image_engagement/report.md` (with a plain-language
verdict), `metrics.json` and `oof_predictions.csv`; the model is saved to
`models/image_engagement.joblib`. `ImageEngagementPredictor` in the same
file loads it for inference, and the Image Analysis page shows its
prediction (with its validated AUC) next to the heuristic once the model
file exists. If the verdict says it doesn't beat chance, delete
`models/image_engagement.joblib` so the page shows the heuristic only.

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
4. Keep this as an additive page in the web app once it exists,
   clearly distinguished from the current heuristic ("Predicted" vs
   "Heuristic" labeling, matching the convention already used
   throughout the UI).

None of this is implemented in the current codebase - do not assume any
Pitt Ads or Instagram data currently influences the web app's
engagement scores.

## Web app pages

**Analyze**
- **Ad Copy** — sentiment + heuristic engagement + explanation + rewrite suggestion for one piece of copy.
- **Image Analysis** — creative-quality score with per-check results (focus, exposure, contrast, noise, resolution, framing), measurements, what to fix, and deterministic style presets (auto-fix, warm tone, punchy, etc.) with re-scored before/after comparison.
- **Bulk Analysis** — same sentiment/engagement pipeline over a CSV of copy, with progress, charts, tables and a CSV download.
- **Website Audit** — SEO / GEO checklist, Core Web Vitals and copy sentiment for one web page (see "Website SEO/GEO analysis").

**Create & Edit**
- **Generative Editing** — Stable Diffusion img2img restyling from a text prompt.
- **Precise Editing** — exact background color swap (`rembg`) and OCR-based text replacement (`easyocr`), for when you need an exact result rather than a generative approximation.
- **Canvas Editor** — draw, line, box, circle, text box and crop tools with undo, all in the browser.

The sidebar also shows which sentiment model is active and which
optional dependencies are currently installed, so you always know what
you're looking at before trusting a number.

## Running tests/checks

```bash
python -m py_compile *.py             # syntax check
python 01_data_preprocessing.py       # should always succeed - see step 2
python 02_model_training.py           # needs data/processed/*.csv from step above
python 03_web_app.py                  # open http://127.0.0.1:8000
```
