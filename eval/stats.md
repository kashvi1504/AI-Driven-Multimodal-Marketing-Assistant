# Statistics: confidence intervals and significance

Generated 2026-11-17 17:40 by `python eval/stats.py`. 95% intervals: bootstrap (10,000 resamples of the ads, seed 0) for copy, Wilson for image rates.

## Ad copy sentiment, held-out test half (90 ads)

| Model | Accuracy [95% CI] | Macro P | Macro R | Macro F1 [95% CI] |
|---|---|---|---|---|
| Fine-tuned DistilBERT (ours) | 76.7% [67.8, 84.4] | 0.81 | 0.77 | 0.77 [0.68, 0.85] |
| Twitter RoBERTa (app default) | 96.7% [92.2, 100.0] | 0.97 | 0.97 | 0.97 [0.92, 1.00] |
| VADER (off-the-shelf baseline) | 73.3% [64.4, 82.2] | 0.75 | 0.73 | 0.74 [0.64, 0.82] |

Per-class precision / recall / F1 [95% CI of F1]:

| Model | Negative | Neutral | Positive |
|---|---|---|---|
| Fine-tuned DistilBERT (ours) | 0.96 / 0.80 / 0.87 [0.76, 0.96] | 0.61 / 0.90 / 0.73 [0.60, 0.83] | 0.86 / 0.60 / 0.71 [0.54, 0.83] |
| Twitter RoBERTa (app default) | 1.00 / 0.93 / 0.97 [0.91, 1.00] | 0.94 / 1.00 / 0.97 [0.91, 1.00] | 0.97 / 0.97 / 0.97 [0.91, 1.00] |
| VADER (off-the-shelf baseline) | 0.92 / 0.77 / 0.84 [0.72, 0.93] | 0.66 / 0.70 / 0.68 [0.53, 0.80] | 0.67 / 0.73 / 0.70 [0.55, 0.81] |

Exact McNemar tests (ads where only one model is right):

- Twitter RoBERTa (app default) vs Fine-tuned DistilBERT (ours): 18 vs 0, p = 7.6e-06
- Twitter RoBERTa (app default) vs VADER (off-the-shelf baseline): 22 vs 1, p = 5.7e-06

## Ad copy sentiment, all 180 ads (supplementary; the dev half was used to choose models and tune rules) (180 ads)

| Model | Accuracy [95% CI] | Macro P | Macro R | Macro F1 [95% CI] |
|---|---|---|---|---|
| Fine-tuned DistilBERT (ours) | 72.8% [66.1, 79.4] | 0.80 | 0.73 | 0.73 [0.66, 0.79] |
| Twitter RoBERTa (app default) | 96.7% [93.9, 98.9] | 0.97 | 0.97 | 0.97 [0.94, 0.99] |
| VADER (off-the-shelf baseline) | 73.3% [66.7, 79.4] | 0.75 | 0.73 | 0.74 [0.67, 0.80] |

Per-class precision / recall / F1 [95% CI of F1]:

| Model | Negative | Neutral | Positive |
|---|---|---|---|
| Fine-tuned DistilBERT (ours) | 0.92 / 0.75 / 0.83 [0.74, 0.90] | 0.57 / 0.92 / 0.70 [0.61, 0.78] | 0.91 / 0.52 / 0.66 [0.54, 0.76] |
| Twitter RoBERTa (app default) | 1.00 / 0.95 / 0.97 [0.94, 1.00] | 0.94 / 0.98 / 0.96 [0.92, 0.99] | 0.97 / 0.97 / 0.97 [0.93, 0.99] |
| VADER (off-the-shelf baseline) | 0.88 / 0.70 / 0.78 [0.68, 0.86] | 0.66 / 0.75 / 0.70 [0.61, 0.79] | 0.70 / 0.75 / 0.73 [0.63, 0.81] |

Exact McNemar tests (ads where only one model is right):

- Twitter RoBERTa (app default) vs Fine-tuned DistilBERT (ours): 43 vs 0, p = 2.3e-13
- Twitter RoBERTa (app default) vs VADER (off-the-shelf baseline): 43 vs 1, p = 5.1e-12

## Copy detectors (90 held-out ads)

| Detector | Positives | Precision | Recall | F1 [95% CI] |
|---|---|---|---|---|
| Call to action | 23 | 0.96 | 0.96 | 0.96 [0.88, 1.00] |
| Urgency | 8 | 0.89 | 1.00 | 0.94 [0.77, 1.00] |

## Images, tuning set

| Measure | k / n | Rate [95% Wilson CI] |
|---|---|---|
| banner | 9 / 9 | 100.0% [70.1, 100.0] |
| blur | 10 / 10 | 100.0% [72.2, 100.0] |
| bright | 10 / 10 | 100.0% [72.2, 100.0] |
| clean | 10 / 10 | 100.0% [72.2, 100.0] |
| clean_lowres | 12 / 12 | 100.0% [75.8, 100.0] |
| dark | 10 / 10 | 100.0% [72.2, 100.0] |
| noise | 8 / 10 | 80.0% [49.0, 94.3] |
| tiny | 10 / 10 | 100.0% [72.2, 100.0] |
| wm_center | 10 / 10 | 100.0% [72.2, 100.0] |
| wm_corner | 10 / 10 | 100.0% [72.2, 100.0] |
| wm_tiled | 8 / 10 | 80.0% [49.0, 94.3] |
| ALL | 107 / 111 | 96.4% [91.1, 98.6] |
| watermark recall | 28 / 30 | 93.3% [78.7, 98.2] |
| watermark false alarms | 0 / 94 | 0.0% [0.0, 3.9] |
| AI recall | 6 / 13 | 46.2% [23.2, 70.9] |
| AI false alarms | 1 / 23 | 4.3% [0.8, 21.0] |

## Images, held-out set

| Measure | k / n | Rate [95% Wilson CI] |
|---|---|---|
| blur | 30 / 32 | 93.8% [79.9, 98.3] |
| bright | 20 / 32 | 62.5% [45.3, 77.1] |
| clean_lowres | 31 / 32 | 96.9% [84.3, 99.4] |
| dark | 32 / 32 | 100.0% [89.3, 100.0] |
| noise | 32 / 32 | 100.0% [89.3, 100.0] |
| tiny | 32 / 32 | 100.0% [89.3, 100.0] |
| wm_center | 32 / 32 | 100.0% [89.3, 100.0] |
| wm_corner | 32 / 32 | 100.0% [89.3, 100.0] |
| wm_tiled | 32 / 32 | 100.0% [89.3, 100.0] |
| ALL | 273 / 288 | 94.8% [91.6, 96.8] |
| watermark recall | 96 / 96 | 100.0% [96.2, 100.0] |
| watermark false alarms | 0 / 192 | 0.0% [0.0, 2.0] |
| AI false alarms | 4 / 32 | 12.5% [5.0, 28.1] |
