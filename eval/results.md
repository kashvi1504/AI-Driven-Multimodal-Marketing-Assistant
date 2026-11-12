# Evaluation results

Generated 2026-11-12 22:29 by `python eval/evaluate.py`.

## Ad copy sentiment (90 hand-labelled ads, `test` split)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Fine-tuned DistilBERT (ours) | 77% | 0.77 |
| Twitter RoBERTa (app default) | 97% | 0.97 |

**Fine-tuned DistilBERT (ours)** confusion (rows = true label, columns = predicted):

| | negative | neutral | positive |
|---|---|---|---|
| negative | 24 | 5 | 1 |
| neutral | 1 | 27 | 2 |
| positive | 0 | 12 | 18 |

**Twitter RoBERTa (app default)** confusion (rows = true label, columns = predicted):

| | negative | neutral | positive |
|---|---|---|---|
| negative | 28 | 1 | 1 |
| neutral | 0 | 30 | 0 |
| positive | 0 | 1 | 29 |

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| Call to action | 96% | 96% | 0.96 |
| Urgency | 89% | 100% | 0.94 |

Average heuristic engagement score by true tone: negative 26.4, neutral 33.1, positive 79.2

Ads the app default still gets wrong:

- *positive → neutral*: Unlock premium for free for 3 months. Sign up today, cancel anytime.
- *negative → neutral*: Due to rising costs, our subscription price will increase from next month.
- *negative → positive*: Our Andheri branch is closing permanently. Thank you for years of support.

## Images (128 images)

Expected-verdict pass rate: **96%** · 5.74 s per image

| Category | Images | Expected | Passed | Mean score | Range |
|---|---|---|---|---|---|
| ai | 13 | no fixed verdict | - | 81.5 | 64-85 |
| banner | 9 | at least Fair (>= 45) | 9 | 81.3 | 75-85 |
| blur | 10 | Needs work (< 45) | 10 | 30.0 | 30-30 |
| bright | 10 | Needs work (< 45) | 10 | 30.2 | 30-32 |
| clean | 10 | Good (>= 70) | 10 | 87.2 | 82-94 |
| clean_lowres | 12 | at least Fair (>= 45) | 12 | 77.0 | 45-85 |
| dark | 10 | Needs work (< 45) | 10 | 30.0 | 30-30 |
| noise | 10 | not Good (< 70) | 8 | 51.3 | 45-76 |
| realworld | 4 | no fixed verdict | - | 55.9 | 37-86 |
| tiny | 10 | Needs work (< 45) | 10 | 30.0 | 30-30 |
| wm_center | 10 | watermark flagged, < 70 | 10 | 40.0 | 40-40 |
| wm_corner | 10 | watermark flagged, < 70 | 10 | 40.0 | 40-40 |
| wm_tiled | 10 | watermark flagged, < 70 | 8 | 49.8 | 40-90 |

**Watermark detection:** flagged 28/30 watermarked images; false alarms on 0/94 images without a watermark.

**AI-generated indicator:** flagged 6/13 AI images; false alarms on 1/23 real images.

Real ads scored as-is: real__bdr283rwynjm1vb2vquginf04qqu.jpg 86 (high); real__bts.jpg 42 (low); real__images_2.jpg 37 (low); real__images.jpg 59 (medium)

Failures: Smart_Banking_Storyboards.pptx_(1)__noise.jpg (noise, 76); Smart_Banking_Storyboards.pptx_(1)__wm_tiled.jpg (wm_tiled, 88); Smart_Banking_Storyboards.pptx__noise.jpg (noise, 76); Smart_Banking_Storyboards.pptx__wm_tiled.jpg (wm_tiled, 90)
