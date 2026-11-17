# Evaluation results

Generated 2026-11-17 15:37 by `python eval/evaluate.py --heldout`.

## Held-out images (thresholds never tuned on these) (288 images)

Expected-verdict pass rate: **94%** · 1.16 s per image

| Category | Images | Expected | Passed | Mean score | Range |
|---|---|---|---|---|---|
| blur | 32 | Needs work (< 45) | 30 | 31.3 | 30-52 |
| bright | 32 | Needs work (< 45) | 20 | 45.0 | 30-85 |
| clean_lowres | 32 | at least Fair (>= 45) | 31 | 77.9 | 30-85 |
| dark | 32 | Needs work (< 45) | 32 | 30.0 | 30-30 |
| noise | 32 | not Good (< 70) | 32 | 44.5 | 30-45 |
| tiny | 32 | Needs work (< 45) | 32 | 30.0 | 30-30 |
| wm_center | 32 | watermark flagged, < 70 | 32 | 39.7 | 30-40 |
| wm_corner | 32 | watermark flagged, < 70 | 31 | 41.1 | 30-85 |
| wm_tiled | 32 | watermark flagged, < 70 | 32 | 39.7 | 30-40 |

**Watermark detection:** flagged 95/96 watermarked images; false alarms on 0/192 images without a watermark.

**AI-generated indicator:** flagged 0/0 AI images; false alarms on 4/32 real images.

Failures: Flower__bright.jpg (bright, 60); Poppy__bright.jpg (bright, 68); Red_Rose__bright.jpg (bright, 85); Sunflower__blur.jpg (blur, 52); Sunflower__bright.jpg (bright, 85); Yellow_Daisy__bright.jpg (bright, 85); Chalk__bright.jpg (bright, 45); Smack__clean.jpg (clean_lowres, 30); Guitar__bright.jpg (bright, 75); Turntable__blur.jpg (blur, 48); Leaf__bright.jpg (bright, 56); Sandollar__wm_corner.jpg (wm_corner, 85); 8ball__bright.jpg (bright, 75); Basketball__bright.jpg (bright, 85); Football__bright.jpg (bright, 45); Tennis__bright.jpg (bright, 78)
