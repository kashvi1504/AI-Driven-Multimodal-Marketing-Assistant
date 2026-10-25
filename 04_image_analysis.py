"""
04_image_analysis.py
----------------------
Image-side quality scoring for marketing visuals.

IMPORTANT - read this before trusting the numbers:
This module is a HEURISTIC, not a model trained on engagement outcomes
(see 10_image_model_training.py for that). It answers "is this image
technically fit to run as an ad?", not "how many likes will it get?".

How an image is scored
  1. Technical checks, each measured at social-feed display size
     (long side <= 1080 px, which is what viewers actually see):
       focus       - no-reference blur metric (Crete-Roffet et al. 2007,
                     "The Blur Effect"): re-blur the image and see how much
                     its edges change. Sharp edges change a lot, blurry
                     ones barely change. Measured on the sharpest region,
                     so a sharp subject on a deliberately soft background
                     isn't punished.
       exposure    - judged from the luminance percentiles, not the mean,
                     so a dark luxury ad or a white-background product
                     shot isn't penalised for its style - only images with
                     no highlights (underexposed) or no shadows (washed out).
       contrast    - tonal range (1st-99th percentile of luminance).
       noise       - estimated sensor/compression grain (Immerkaer 1996).
       resolution  - shortest side in pixels.
       framing     - closeness to a standard ad aspect ratio.
     Colour saturation is reported but NOT scored: muted, monochrome or
     black-and-white ads are a style choice, not a defect.
  2. Critical defects cap the score. Blur, bad exposure and very low
     resolution can't be averaged away by good scores elsewhere, so a
     badly blurred image is always rated "low"; heavy grain or fog caps
     it at "medium" (SCORE_CAPS).
  3. CLIP zero-shot appeal (optional, 30% weight): a real pretrained model
     comparing the image with medium-neutral "polished ad" vs "poor image"
     descriptions. Not validated against engagement data.

Treat the output as a quality check with explanations, not a prediction
of real engagement.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat
from scipy.ndimage import convolve, uniform_filter1d


# ── CLIP zero-shot scorer ─────────────────────────────────────────────
class ClipScorer:
    """Loads openai/clip-vit-base-patch32 via transformers (already a
    project dependency - no extra install needed). Scores an image
    against paired "polished ad" vs "poor image" descriptions.

    Prompts are worded to be neutral about the medium (photo, graphic,
    illustration) so flat-design and text-led ads aren't marked down for
    not being photographs.
    """

    MODEL_NAME = "openai/clip-vit-base-patch32"

    # POSITIVE_PROMPTS[i] is compared against NEGATIVE_PROMPTS[i]
    POSITIVE_PROMPTS = [
        "a sharp, crisp, in-focus image",
        "a polished, professional advertisement",
        "a clean, well-designed social media post",
        "a well-lit, high-quality image",
    ]
    NEGATIVE_PROMPTS = [
        "a blurry, out-of-focus image",
        "an amateur, low-quality advertisement",
        "a cluttered, badly designed social media post",
        "a dark, grainy, low-quality image",
    ]

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model.eval()

    def score(self, image: Image.Image) -> dict:
        import torch

        prompts = self.POSITIVE_PROMPTS + self.NEGATIVE_PROMPTS
        inputs = self.processor(text=prompts, images=image.convert("RGB"), return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits_per_image[0].numpy()
        return {"clip_score": clip_score_from_logits(logits)}


def clip_score_from_logits(logits: np.ndarray) -> float:
    """0-100: average over prompt pairs of P(positive | positive vs negative).
    Pairwise comparison is steadier than one softmax over all prompts,
    where a single prompt tends to take all the probability mass."""
    n = len(ClipScorer.POSITIVE_PROMPTS)
    pos, neg = np.asarray(logits[:n], dtype=float), np.asarray(logits[n:2 * n], dtype=float)
    return float(100 * np.mean(1 / (1 + np.exp(neg - pos))))


# ── Measurable visual features (no model needed, fast) ────────────────
DISPLAY_SIZE = 1080  # long side, px - roughly what a social feed shows
AD_RATIOS = {"1:1": 1.0, "4:5": 0.8, "3:4": 0.75, "2:3": 2 / 3, "9:16": 9 / 16,
             "4:3": 4 / 3, "16:9": 16 / 9, "1.91:1": 1.91}


def _display_gray(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    scale = DISPLAY_SIZE / max(gray.size)
    if scale < 1:
        gray = gray.resize((max(1, round(gray.width * scale)), max(1, round(gray.height * scale))), Image.LANCZOS)
    return np.asarray(gray, dtype=float)


def _reblur(gray: np.ndarray) -> tuple[float, float]:
    """Crete-Roffet blur metric: 0 = perfectly sharp, 1 = completely blurred.
    Also returns the edge energy, used to skip flat (textureless) regions."""
    blur, energy = 0.0, 0.0
    for axis in (0, 1):
        if gray.shape[axis] < 3:
            return 1.0, 0.0
        diff = np.abs(np.diff(gray, axis=axis))
        diff_blurred = np.abs(np.diff(uniform_filter1d(gray, 9, axis=axis), axis=axis))
        total = diff.sum()
        lost = np.maximum(0, diff - diff_blurred).sum()  # detail the re-blur destroyed
        blur = max(blur, (total - lost) / total if total > 0 else 1.0)
        energy += diff.mean()
    return float(blur), float(energy / 2)


def _focus_blur(gray: np.ndarray, grid: int = 6) -> float:
    """Blur of the sharpest ~10% of textured tiles - i.e. "is anything in
    focus?". A fully blurred image is blurry in every tile; a sharp product
    on a soft background still has sharp tiles."""
    h, w = gray.shape
    tiles = []
    for i in range(grid):
        for j in range(grid):
            tile = gray[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
            blur, energy = _reblur(tile)
            if energy > 1.0:  # skip flat areas (sky, plain backgrounds)
                tiles.append(blur)
    if len(tiles) < 2:
        return _reblur(gray)[0]
    return float(np.percentile(tiles, 10))


def _noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise estimate: std-dev of grain in 0-255 units.
    Pure black/white areas are skipped - clipping hides grain there, so
    including them would make a grainy image on a white background look clean."""
    if min(gray.shape) < 3:
        return 0.0
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=float)
    response = np.abs(convolve(gray, mask)[1:-1, 1:-1])
    unclipped = ((gray > 5) & (gray < 250))[1:-1, 1:-1]
    if unclipped.mean() < 0.01:
        return 0.0
    return float(np.sqrt(np.pi / 2) * response[unclipped].mean() / 6)


def extract_visual_features(image: Image.Image) -> dict:
    rgb = image.convert("RGB")
    gray = _display_gray(rgb)
    p1, p99 = np.percentile(gray, [1, 99])
    width, height = rgb.size

    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "brightness": float(gray.mean()),        # mean luminance, 0-255
        "shadows": float(p1),                    # 1st percentile luminance (darkest tones)
        "highlights": float(p99),                # 99th percentile luminance (brightest tones)
        "dynamic_range": float(p99 - p1),        # tonal range, 0-255
        "saturation": ImageStat.Stat(rgb.convert("HSV")).mean[1],  # 0-255, reported only
        "blur": _focus_blur(gray),               # 0 sharp - 1 blurred (sharpest region)
        "blur_global": _reblur(gray)[0],         # same metric over the whole frame
        "noise": _noise_sigma(gray),             # grain std-dev, 0-255
    }


def _ramp(value: float, bad: float, good: float) -> float:
    """0 at `bad`, 1 at `good`, linear in between (works in either direction)."""
    return float(np.clip((value - bad) / (good - bad), 0.0, 1.0))


def _status(score: float) -> str:
    return "pass" if score >= 0.8 else "warn" if score >= 0.4 else "fail"


# name -> weight in the technical score.
CHECK_WEIGHTS = {"focus": 0.35, "exposure": 0.20, "contrast": 0.15, "resolution": 0.15, "noise": 0.10, "framing": 0.05}
# Defects that can't be averaged away: name -> (cap at score 0, cap at score 100).
# Blur, bad exposure or a tiny image cap the overall score at 30 ("low");
# heavy grain or fog cap it at 45 (bottom of "medium").
SCORE_CAPS = {"focus": (30, 100), "exposure": (30, 100), "resolution": (30, 100),
              "contrast": (45, 100), "noise": (45, 100)}
CRITICAL_CHECKS = ("focus", "exposure", "resolution")  # the ones that can force a "low" rating


def score_visual_features(features: dict) -> dict:
    """Turns raw measurements into per-check scores (0-100), a pass/warn/fail
    status and a plain-language message for each. Thresholds were calibrated
    on photos and graphic ads at controlled blur/noise/exposure levels."""
    f = features
    short_side = min(f["width"], f["height"])
    ratio_name, ratio = min(AD_RATIOS.items(), key=lambda kv: abs(np.log(f["aspect_ratio"] / kv[1])))
    ratio_off = abs(np.log(f["aspect_ratio"] / ratio))

    raw = {
        # sharpest-region blur: in-focus images measure ~0.1-0.31 (photos and
        # graphics alike); a 2px Gaussian blur pushes them to ~0.46-0.65
        "focus": _ramp(f["blur"], 0.55, 0.32),
        # underexposed = even the brightest parts are dark; washed out = even the darkest are light
        "exposure": min(_ramp(f["highlights"], 80, 140), _ramp(f["shadows"], 200, 140)),
        "contrast": _ramp(f["dynamic_range"], 40, 110),
        "resolution": _ramp(short_side, 200, 600),
        "noise": _ramp(f["noise"], 15, 6),
        "framing": _ramp(ratio_off, 0.25, 0.03),
    }

    under = f["highlights"] < 140
    messages = {
        "focus": {
            "pass": "In focus - edges are crisp.",
            "warn": "Slightly soft - the subject isn't fully crisp. Use a sharper original or refocus.",
            "fail": "Blurry - nothing in the frame is in focus. Reshoot or use a sharper original; "
                    "sharpening can't recover this.",
        },
        "exposure": {
            "pass": "Exposure is fine - there are clear highlights and shadows.",
            "warn": "Exposure is a bit " + ("dark" if under else "washed out") + " - "
                    + ("brighten the subject." if under else "add some depth to the darker tones."),
            "fail": ("Underexposed - even the brightest areas are dark." if under
                     else "Washed out - there are no real shadows anywhere."),
        },
        "contrast": {
            "pass": "Good tonal range.",
            "warn": "Fairly flat - the image may look hazy in a busy feed.",
            "fail": "Very low contrast - the image looks foggy and text/products won't stand out.",
        },
        "resolution": {
            "pass": f"Resolution is fine ({f['width']}x{f['height']}).",
            "warn": f"Low resolution ({f['width']}x{f['height']}) - feeds display ~1080px wide; "
                    "it may look soft when enlarged.",
            "fail": f"Very low resolution ({f['width']}x{f['height']}) - it will look pixelated in a feed.",
        },
        "noise": {
            "pass": "Clean - no visible grain.",
            "warn": "Some visible grain or compression noise.",
            "fail": "Very grainy/noisy - likely shot in low light or heavily compressed.",
        },
        "framing": {
            "pass": f"Aspect ratio matches the standard {ratio_name} ad format.",
            "warn": f"Close to {ratio_name}, but not exact - crop to {ratio_name} to avoid platform cropping.",
            "fail": f"Non-standard aspect ratio - crop to a standard ad format such as 1:1, 4:5 or 9:16 "
                    f"(closest: {ratio_name}).",
        },
    }

    checks = []
    for name, s in raw.items():
        status = _status(s)
        checks.append({"name": name, "score": 100 * s, "status": status, "message": messages[name][status],
                       "critical": name in CRITICAL_CHECKS, "weight": CHECK_WEIGHTS[name]})

    technical = 100 * sum(raw[k] * w for k, w in CHECK_WEIGHTS.items()) / sum(CHECK_WEIGHTS.values())
    caps = {k: lo + (hi - lo) * raw[k] for k, (lo, hi) in SCORE_CAPS.items()}
    cap_name = min(caps, key=caps.get)

    reasons = [c["message"] for c in sorted(checks, key=lambda c: c["score"]) if c["status"] != "pass"]
    if not reasons:
        reasons.append("Focus, exposure, contrast, noise, resolution and framing all look solid.")
    if f["saturation"] < 25:
        reasons.append("Note: colours are very muted or monochrome. That's fine if it's intentional - "
                       "it isn't scored.")

    return {
        "feature_score": technical,
        "cap": caps[cap_name],
        "cap_check": cap_name,
        "checks": checks,
        "sub_scores": {c["name"]: c["score"] for c in checks},
        "reasons": reasons,
    }


# ── Combined scorer ────────────────────────────────────────────────────
CLIP_WEIGHT = 0.3


def combine_scores(visual: dict, clip_score: "float | None") -> dict:
    """Blends the technical score with CLIP appeal, then applies the cap from
    the worst critical check (SCORE_CAPS), so a blurry or badly exposed image
    can't be rated well no matter how good the rest is."""
    base = visual["feature_score"]
    if clip_score is not None:
        base = (1 - CLIP_WEIGHT) * base + CLIP_WEIGHT * clip_score
    cap = visual["cap"]
    overall = min(base, cap)
    return {"score": overall, "capped": cap < base, "cap": cap}


def score_label(score: float) -> str:
    return "high" if score >= 70 else "medium" if score >= 45 else "low"


def score_image(image: Image.Image, clip_scorer: "ClipScorer | None" = None) -> dict:
    """Combines the technical checks with CLIP appeal (if available) into
    one 0-100 quality score + per-check explanations."""
    features = extract_visual_features(image)
    visual = score_visual_features(features)

    clip_score = None
    if clip_scorer is not None:
        try:
            clip_score = clip_scorer.score(image)["clip_score"]
        except Exception:  # noqa: BLE001 - fall back to technical checks only
            clip_score = None

    combined = combine_scores(visual, clip_score)
    reasons = list(visual["reasons"])
    if combined["capped"]:
        reasons.insert(0, f"Score capped at {combined['cap']:.0f} by the {visual['cap_check']} check - "
                          "a defect this serious can't be offset by strengths elsewhere.")

    return {
        "score": combined["score"],
        "label": score_label(combined["score"]),
        "used_clip": clip_score is not None,
        "clip_score": clip_score,
        "feature_score": visual["feature_score"],
        "capped": combined["capped"],
        "cap_check": visual["cap_check"] if combined["capped"] else None,
        "checks": visual["checks"],
        "sub_scores": visual["sub_scores"],
        "reasons": reasons,
        "features": features,
    }


# ── Refinement: real pixel adjustments, not regeneration ──────────────
def _tint(image: Image.Image, color: tuple, alpha: float) -> Image.Image:
    """Blends a solid color over the image at low opacity - the standard
    trick for a warm/cool color-grade look without needing anything generative."""
    overlay = Image.new("RGB", image.size, color)
    return Image.blend(image, overlay, alpha)


def _crop_to_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    w, h = image.size
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return image.crop((left, 0, left + new_w, h))
    new_h = int(w / target_ratio)
    top = (h - new_h) // 2
    return image.crop((0, top, w, top + new_h))


def style_auto_fix(image: Image.Image) -> Image.Image:
    """Only corrects measured issues (brightness/contrast/saturation/
    sharpness/aspect) - the smallest possible edit."""
    return refine_image(image)


def style_warm(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    out = ImageEnhance.Color(rgb).enhance(1.3)
    out = ImageEnhance.Brightness(out).enhance(1.1)
    return _tint(out, (255, 150, 60), 0.22)


def style_punchy(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    out = ImageEnhance.Color(rgb).enhance(1.7)
    out = ImageEnhance.Contrast(out).enhance(1.4)
    return out


def style_high_contrast(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    out = ImageEnhance.Contrast(rgb).enhance(1.55)
    out = ImageEnhance.Sharpness(out).enhance(1.4)
    return out


def style_bright_airy(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    out = ImageEnhance.Brightness(rgb).enhance(1.3)
    out = ImageEnhance.Contrast(out).enhance(0.85)
    out = ImageEnhance.Color(out).enhance(0.85)
    return out


def style_moody(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    out = ImageEnhance.Contrast(rgb).enhance(1.35)
    out = ImageEnhance.Color(out).enhance(0.7)
    out = ImageEnhance.Brightness(out).enhance(0.85)
    return _tint(out, (40, 70, 130), 0.15)


def style_square_crop(image: Image.Image) -> Image.Image:
    return _crop_to_ratio(image.convert("RGB"), 1.0)


def style_vertical_crop(image: Image.Image) -> Image.Image:
    return _crop_to_ratio(image.convert("RGB"), 9 / 16)


STYLE_PRESETS = {
    "Auto-fix issues": style_auto_fix,
    "Warm tone": style_warm,
    "Punchy / vibrant": style_punchy,
    "High contrast": style_high_contrast,
    "Bright & airy": style_bright_airy,
    "Moody / cinematic": style_moody,
    "Square crop (1:1)": style_square_crop,
    "Vertical crop (9:16)": style_vertical_crop,
}


def refine_image(image: Image.Image) -> Image.Image:
    """Corrects only what the checks flag: crops toward the nearest standard
    ad ratio, fixes exposure and flat contrast, reduces heavy grain and
    gently sharpens slightly soft images. Real blur can't be recovered, so
    a "fail" on focus is left for the reasons to explain.
    Fast, deterministic, doesn't require any generative model."""
    rgb = image.convert("RGB")
    checks = {c["name"]: c for c in score_visual_features(extract_visual_features(rgb))["checks"]}

    if checks["framing"]["status"] != "pass":
        f = extract_visual_features(rgb)
        target = min(AD_RATIOS.values(), key=lambda r: abs(np.log(f["aspect_ratio"] / r)))
        rgb = _crop_to_ratio(rgb, target)

    features = extract_visual_features(rgb)
    out = rgb
    if checks["noise"]["status"] != "pass":
        out = out.filter(ImageFilter.MedianFilter(3))

    if features["highlights"] < 140:
        out = ImageEnhance.Brightness(out).enhance(min(2.5, 200 / max(features["highlights"], 1)))
    if checks["contrast"]["status"] != "pass" or features["shadows"] > 140:
        out = ImageOps.autocontrast(out, cutoff=1)

    if checks["focus"]["status"] == "warn":
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))

    return out
