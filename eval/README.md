# Accuracy testing

Built after the November 2026 review, which asked for more accurate copy and
image analysis, testing across a wide range of photos and ad copies, AI-image
and watermark checks, a likes : followers check, and the scope and speed of
Bulk Analysis. Full numbers from the last run are in `results.md`;
Bulk Analysis timings are in `bulk_benchmark.md`.

## Test sets

**Ad copy** - `copy_test_set.csv`, written by `build_copy_test_set.py`.
180 ads written and labelled by hand: 60 positive (offers, launches, festive
sales such as Diwali and Holi, emoji-heavy social posts), 60 neutral (store
hours, specs, policies) and 60 negative (apologies, outages, recalls, price
rises). Each row also marks whether it really contains a call to action and
real urgency. Rows alternate between a **dev** half, the only half used while
choosing models and tuning rules, and a **test** half, used only for the final
numbers below.

**Images** - built by `build_image_test_set.py` into `data/eval/` (not
committed: some source photos are personal). 128 images:

- 10 clean, high-resolution originals (camera photos, real ads, a graphic) and
  12 smaller clean originals;
- from each high-resolution original, versions with a known defect: blurred,
  shrunk to 180 px, underexposed, overexposed, heavy grain, three styles of
  stock-photo watermark (tiled "shutterstock", centred "Getty Images" with a
  credit bar, corner "iStock by Getty Images"), and a 3.5 : 1 banner crop;
- 13 AI-generated images: 12 made with the local Stable Diffusion 1.5 model
  (product, food, interior and lifestyle ad prompts) and one ChatGPT image;
- 4 real ads scored as they are (no fixed verdict).

Because each defect is added on purpose, the correct verdict for every image
is known. Limitation: the image thresholds were tuned on this same set
(there was no separate image test half), so new kinds of images should be
added as they come up.

## Results - before and after

### Ad copy (held-out test half, 90 ads)

| | Before | After |
|---|---|---|
| Sentiment accuracy | 77% (our fine-tuned DistilBERT) | **97%** (Twitter RoBERTa) |
| Positive ads called "neutral" | 12 of 30 | 1 of 30 |
| Call-to-action detection (F1) | 0.72 * | **0.96** |
| Urgency detection (F1) | 0.26 * | **0.94** |
| Avg engagement score: negative / neutral / positive | - | 26 / 33 / 79 |

\* measured on the dev half before the rules were rewritten.

Why the sentiment model changed: our DistilBERT was fine-tuned on general text
(it scores 86% on its own test set of movie-review phrases and tweets) and
reads sales language like "Grab yours now" or "won't last long" as neutral.
The public `cardiffnlp/twitter-roberta-base-sentiment-latest` model, trained on
~124M tweets, reads emojis and promotional slang correctly. On the old general
test set it scores lower (61%), because most of that set is movie-review
fragments ("bawdy", "an epic rather than") that look nothing like ads. The
app analyses ads, so it now uses RoBERTa; `COPY_SENTIMENT_MODEL=fine-tuned`
switches back.

The 3 ads it still gets wrong are genuinely mixed in tone ("free for 3
months... cancel anytime", a polite price-rise notice, a branch closing with a
thank-you).

What changed in the rules: calls to action are now command verbs at the start
of a phrase ("Grab yours", "Book now", "Sign up") instead of 10 fixed words, so
"grab", "order", "discover" and "treat yourself" count. Urgency now means a
real time limit or scarcity ("ends Sunday", "only 10 left", "won't last long",
"while stocks last") - a bare "now" or "only" no longer counts.

### Images (128 images)

| | Before | After |
|---|---|---|
| Expected verdict met | 55% | **96%** |
| Blurred / tiny / underexposed caught | 30 / 30 | 30 / 30 |
| Overexposed caught | 0 / 10 | **10 / 10** |
| Heavy grain kept below "Good" | 1 / 10 | **8 / 10** |
| Clean images still rated Good / Fair | 22 / 22 | 22 / 22 |
| Stock watermark flagged | 0 / 30 (no check) | **28 / 30**, with 0 false alarms on 94 unwatermarked images |
| AI-generated flagged | no check | **6 / 13**, with 1 false alarm on 23 real images |

What failed and why:

- Grain and the tiled watermark on the two **storyboard graphics**: they are
  mostly flat white with small text, so there is little area to measure
  grain in, and the OCR mixes the watermark letters up with the storyboard's
  own small text.
- **AI detection** misses about half the Stable Diffusion images. The metadata
  check caught the ChatGPT image (its Content Credentials say "OpenAI" /
  "trainedAlgorithmicMedia") even though the pixel model gave it only 19%. A
  second model, `Ateeqq/ai-vs-human-image-detector`, caught 8 / 13 but wrongly
  flagged 12 / 23 real photos, so it was rejected. This is why the page calls
  it an indicator, not proof.

### Watermarked images - what result do they get?

Image analysis gives a quality score, not positive / neutral / negative. A
watermarked image is flagged and capped at **40 / 100 ("Needs work")** with the
message to license the image or use your own, because a stock preview can't
legally run as an ad.

### Bulk Analysis scope and speed

There is no model limit on rows; uploads are capped at 20,000 rows
(`BULK_MAX_ROWS`) only to protect memory. On this Mac (Apple silicon):

| | Before | After |
|---|---|---|
| 1,000 rows, analysis only | 35.5 s | **4.3 s** |
| 5,000 rows, analysis only | 168 s | **23.5 s** |
| 1,000 rows with rewrites | ~18 min | **3 min 8 s** |

Speed-ups: models on the Apple-silicon GPU (MPS), batched sentiment, rows that
already score 70+ are not rewritten, identical rows are rewritten once, and
rewrites for 8 rows are generated in one model call. Every rewrite still passes
the grounding check (154 invented-claim candidates rejected in the 1,000-row
run).

## Re-running

```bash
python eval/build_copy_test_set.py
python eval/build_image_test_set.py --sources <clean images> --realworld <real ads>
python eval/evaluate.py            # --split dev for the tuning half
python eval/benchmark_bulk.py
```
