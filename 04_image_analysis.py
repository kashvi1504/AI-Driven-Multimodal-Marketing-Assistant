"""
04_image_analysis.py
----------------------
Image-side engagement scoring for marketing visuals.

IMPORTANT - read this before trusting the numbers:
Unlike the text sentiment model (trained + validated on ~17k real labeled
examples, ~85% test accuracy), there is no dataset available that pairs
real marketing images with real engagement outcomes at a usable scale.
So this module is a HEURISTIC, not a trained/validated model:

  - CLIP zero-shot similarity: a real, pretrained, widely-used model,
    but not validated against YOUR engagement data - it's checking how
    well an image matches general "engaging marketing photo" concepts.
  - Visual features (brightness/contrast/saturation/sharpness/aspect
    ratio): grounded in general photography/social-media conventions,
    not learned from labeled outcomes.

Treat the output as an informed heuristic for direction (better/worse),
not a validated percentage prediction of real engagement.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat


# ── CLIP zero-shot scorer ─────────────────────────────────────────────
class ClipScorer:
    """Loads openai/clip-vit-base-patch32 via transformers (already a
    project dependency - no extra install needed). Scores an image
    against curated "good marketing photo" vs "poor photo" concepts.
    """

    MODEL_NAME = "openai/clip-vit-base-patch32"

    POSITIVE_PROMPTS = [
        "a vibrant, professional marketing photograph",
        "a bright, high-quality product photo",
        "an eye-catching, well-composed advertisement image",
        "a clean, well-lit social media photo",
    ]
    NEGATIVE_PROMPTS = [
        "a dark, blurry, low-quality photo",
        "a dull, poorly lit amateur snapshot",
        "a cluttered, badly composed image",
        "an out-of-focus, grainy photo",
    ]

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model.eval()

    def score(self, image: Image.Image) -> dict:
        import torch

        prompts = self.POSITIVE_PROMPTS + self.NEGATIVE_PROMPTS
        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].tolist()

        n_pos = len(self.POSITIVE_PROMPTS)
        positive_mass = sum(probs[:n_pos])
        negative_mass = sum(probs[n_pos:])
        # normalise to a 0-100 "semantic quality" score
        clip_score = 100 * positive_mass / (positive_mass + negative_mass + 1e-9)
        return {"clip_score": clip_score, "positive_mass": positive_mass, "negative_mass": negative_mass}


# ── Measurable visual features (no model needed, fast) ────────────────
def extract_visual_features(image: Image.Image) -> dict:
    rgb = image.convert("RGB")
    hsv = rgb.convert("HSV")
    stat_rgb = ImageStat.Stat(rgb)
    stat_hsv = ImageStat.Stat(hsv)

    # brightness: mean pixel value across channels, 0-255
    brightness = sum(stat_rgb.mean) / 3

    # contrast: mean of per-channel stddev, 0-255
    contrast = sum(stat_rgb.stddev) / 3

    # saturation: mean of the S channel in HSV, 0-255
    saturation = stat_hsv.mean[1]

    # sharpness proxy: variance of an edge-detected version - blurry
    # images have low edge variance, sharp images have high variance
    edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES)
    sharpness = float(np.array(edges).var())

    width, height = rgb.size
    aspect_ratio = width / height

    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "sharpness": sharpness,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
    }


def _score_in_range(value: float, low: float, high: float, hard_low: float, hard_high: float) -> float:
    """Returns 1.0 inside [low, high], decaying linearly to 0 at hard_low/hard_high."""
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, (value - hard_low) / (low - hard_low)) if low > hard_low else 0.0
    return max(0.0, (hard_high - value) / (hard_high - high)) if hard_high > high else 0.0


def score_visual_features(features: dict) -> dict:
    """Turns raw measurements into a 0-100 score with reasons. Ranges
    are common photography/social-media guidance, not learned from data."""
    reasons = []

    brightness_score = _score_in_range(features["brightness"], 90, 190, 30, 240)
    if brightness_score < 0.5:
        reasons.append("Image is quite dark or overexposed - aim for a mid-range brightness.")

    contrast_score = _score_in_range(features["contrast"], 40, 90, 10, 120)
    if contrast_score < 0.5:
        reasons.append("Low contrast - the image may look flat or washed out.")

    saturation_score = _score_in_range(features["saturation"], 60, 160, 10, 220)
    if saturation_score < 0.5:
        reasons.append("Colors look muted - a bit more saturation often helps ads stand out.")

    sharpness_score = min(1.0, features["sharpness"] / 800)
    if sharpness_score < 0.5:
        reasons.append("Image may be soft/blurry - sharper images tend to perform better.")

    # common, well-supported social aspect ratios: 1:1, 4:5, 9:16, 16:9
    common_ratios = [1.0, 0.8, 9 / 16, 16 / 9]
    closest_diff = min(abs(features["aspect_ratio"] - r) for r in common_ratios)
    aspect_score = max(0.0, 1.0 - closest_diff / 0.5)
    if aspect_score < 0.5:
        reasons.append("Unusual aspect ratio - square (1:1), portrait (4:5), or vertical (9:16) tend to perform best.")

    if not reasons:
        reasons.append("Brightness, contrast, saturation, sharpness, and aspect ratio all look solid.")

    overall = 100 * np.mean([brightness_score, contrast_score, saturation_score, sharpness_score, aspect_score])
    return {
        "feature_score": overall,
        "sub_scores": {
            "brightness": brightness_score * 100,
            "contrast": contrast_score * 100,
            "saturation": saturation_score * 100,
            "sharpness": sharpness_score * 100,
            "aspect_ratio": aspect_score * 100,
        },
        "reasons": reasons,
    }


# ── Combined scorer ────────────────────────────────────────────────────
def score_image(image: Image.Image, clip_scorer: "ClipScorer | None" = None) -> dict:
    """Combines CLIP semantic score (if available) with visual features
    into one 0-100 heuristic engagement score + human-readable reasons."""
    features = extract_visual_features(image)
    visual = score_visual_features(features)

    if clip_scorer is not None:
        try:
            clip_result = clip_scorer.score(image)
            overall = 0.5 * clip_result["clip_score"] + 0.5 * visual["feature_score"]
            used_clip = True
        except Exception:
            overall = visual["feature_score"]
            clip_result = None
            used_clip = False
    else:
        overall = visual["feature_score"]
        clip_result = None
        used_clip = False

    label = "high" if overall >= 70 else "medium" if overall >= 45 else "low"
    return {
        "score": overall,
        "label": label,
        "used_clip": used_clip,
        "clip_score": clip_result["clip_score"] if clip_result else None,
        "feature_score": visual["feature_score"],
        "sub_scores": visual["sub_scores"],
        "reasons": visual["reasons"],
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
    """Nudges brightness/contrast/color/sharpness toward the target
    ranges used in score_visual_features, and center-crops toward the
    nearest common social aspect ratio if the original is far off.
    Fast, deterministic, doesn't require any generative model."""
    rgb = image.convert("RGB")
    features = extract_visual_features(rgb)

    # aspect ratio: center-crop toward the closest common ratio if it's
    # notably off (keeps refine_image consistent with what the reasons say)
    common_ratios = {"1:1": 1.0, "4:5": 0.8, "9:16": 9 / 16, "16:9": 16 / 9}
    current = features["aspect_ratio"]
    closest_name, closest_ratio = min(common_ratios.items(), key=lambda kv: abs(current - kv[1]))
    if abs(current - closest_ratio) > 0.15:
        w, h = rgb.size
        target_ratio = closest_ratio
        if w / h > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            rgb = rgb.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            rgb = rgb.crop((0, top, w, top + new_h))
        features = extract_visual_features(rgb)  # re-measure after crop

    out = rgb
    if features["brightness"] < 90:
        out = ImageEnhance.Brightness(out).enhance(1.15)
    elif features["brightness"] > 190:
        out = ImageEnhance.Brightness(out).enhance(0.9)

    if features["contrast"] < 40:
        out = ImageEnhance.Contrast(out).enhance(1.2)

    if features["saturation"] < 60:
        out = ImageEnhance.Color(out).enhance(1.25)

    if features["sharpness"] < 400:
        out = out.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))

    return out