"""
06_image_editing.py
---------------------
Precise, targeted image editing - background color swap and text
replacement. Deliberately NOT using Stable Diffusion here: diffusion is
probabilistic (approximate colors, unreliable text rendering), while
these tasks need exact control. Segmentation + compositing gives an
exact color; OCR + redraw gives exact text control. Both run in
seconds on CPU - no GPU needed.

Two capabilities:
  1. Background swap: cleanly separates the subject from the
     background (rembg) and composites it onto any solid color you pick.
  2. Text replacement: detects text regions (easyocr) so you can pick
     one and replace it with new text, redrawn in roughly the same spot.

First run downloads small model files (~250MB total for both models,
much smaller than a diffusion model) - one-time, then cached.
"""

from PIL import Image, ImageDraw, ImageFont


# ── Background removal + color swap ───────────────────────────────────
def remove_background(image: Image.Image) -> Image.Image:
    """Returns an RGBA image with the background made transparent.
    Uses rembg (U^2-Net under the hood) - downloads its model on first use."""
    from rembg import remove
    return remove(image.convert("RGB"))


def composite_on_color(foreground_rgba: Image.Image, color: tuple) -> Image.Image:
    """Places the (already background-removed) subject onto a flat
    color background. `color` is an (R, G, B) tuple."""
    background = Image.new("RGBA", foreground_rgba.size, color + (255,))
    composite = Image.alpha_composite(background, foreground_rgba)
    return composite.convert("RGB")


def swap_background_to_color(image: Image.Image, color: tuple) -> Image.Image:
    """One-call convenience: remove background, then place subject on
    the given solid color."""
    cutout = remove_background(image)
    return composite_on_color(cutout, color)


# ── Text detection + replacement ───────────────────────────────────────
def detect_text_regions(image: Image.Image) -> list:
    """Returns a list of {bbox: (x1,y1,x2,y2), text: str, confidence: float}
    for each text region found. Uses easyocr - downloads its detection/
    recognition models on first use (~100MB)."""
    import numpy as np
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(np.array(image.convert("RGB")))

    regions = []
    for box, text, confidence in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        regions.append({
            "bbox": (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
            "text": text,
            "confidence": float(confidence),
        })
    return regions


def _sample_fill_color(image: Image.Image, bbox: tuple) -> tuple:
    """Samples color from thin strips just OUTSIDE the box (never inside
    it) on all four sides, then takes the per-channel MEDIAN rather than
    mean. Median matters here: on busy layouts (e.g. another block of
    text sitting close to the one being replaced), a plain average gets
    dragged toward whatever bright/dark pixels are nearby; the median is
    far more robust to that kind of contamination."""
    import statistics

    x1, y1, x2, y2 = bbox
    w, h = image.size
    strip = max(6, int((y2 - y1) * 0.25))

    regions = [
        image.crop((max(0, x1 - strip), max(0, y1 - strip), min(w, x2 + strip), y1)),  # above
        image.crop((max(0, x1 - strip), y2, min(w, x2 + strip), min(h, y2 + strip))),  # below
        image.crop((max(0, x1 - strip), y1, x1, y2)),  # left
        image.crop((x2, y1, min(w, x2 + strip), y2)),  # right
    ]

    pixels = []
    for region in regions:
        if region.size[0] > 0 and region.size[1] > 0:
            pixels.extend(list(region.convert("RGB").getdata()))

    if not pixels:
        return (255, 255, 255)

    r = int(statistics.median(p[0] for p in pixels))
    g = int(statistics.median(p[1] for p in pixels))
    b = int(statistics.median(p[2] for p in pixels))
    return (r, g, b)


def replace_text(
    image: Image.Image,
    bbox: tuple,
    new_text: str,
    text_color: tuple = (0, 0, 0),
    font_path: str | None = None,
    fill_color: tuple | None = None,
) -> Image.Image:
    """Erases the old text in bbox (fills with a sampled nearby color,
    or a manually supplied fill_color if auto-sampling doesn't work
    well for this image) and draws new_text centered in the same spot.
    This is a heuristic replacement, not true generative inpainting -
    works best on flat, simple backgrounds; busy layouts (text close to
    other text/graphics) are the hard case - use fill_color to override
    if the automatic sample looks wrong."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)

    resolved_fill = fill_color if fill_color is not None else _sample_fill_color(out, bbox)
    draw.rectangle(bbox, fill=resolved_fill)

    x1, y1, x2, y2 = bbox
    box_h = y2 - y1
    box_w = x2 - x1

    font_size = max(10, int(box_h * 0.75))
    font = None
    while font_size > 8:
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(size=font_size)
        except Exception:
            font = ImageFont.load_default()
            break
        text_w = draw.textlength(new_text, font=font)
        if text_w <= box_w * 0.95:
            break
        font_size -= 2

    text_w = draw.textlength(new_text, font=font)
    text_x = x1 + max(0, (box_w - text_w) / 2)
    text_y = y1 + max(0, (box_h - font_size) / 2)

    # a slight stroke makes replacement text read as bolder/more
    # legible, closer to the heavy display fonts typical in ad banners
    draw.text(
        (text_x, text_y), new_text, fill=text_color, font=font,
        stroke_width=max(1, font_size // 30), stroke_fill=text_color,
    )
    return out