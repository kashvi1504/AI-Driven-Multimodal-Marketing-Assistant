"""
eval/build_image_test_set.py
----------------------------
Builds the image test set in data/eval/images/ (git-ignored) plus
data/eval/image_manifest.csv, for python eval/evaluate.py.

Clean source images are copied in, then defect versions are made from
each high-resolution one, so every image's correct verdict is known:

  clean        -> should score Good (>= 70)
  clean_lowres -> small originals (~500 px): at least Fair (>= 45)
  blur, tiny, dark, bright -> serious defect: Needs work (< 45)
  noise        -> heavy grain: not Good (< 70)
  wm_tiled / wm_center / wm_corner -> stock-photo watermark: flagged, not Good
  banner       -> odd 3.5:1 crop: framing warning, still >= 45
  ai           -> made by an image generator: flagged as likely AI

    python eval/build_image_test_set.py --sources ~/Downloads/eval_sources

--sources is a folder of your own clean images (none are committed); the
script also uses Apple's sample pictures on a Mac when they exist, and the
Stable Diffusion images in data/eval/ai_generated/ for the "ai" category.
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR  # noqa: E402

EVAL_DIR = os.path.join(DATA_DIR, "eval")
OUT = os.path.join(EVAL_DIR, "images")
APPLE = "/Library/User Pictures"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic")


def load(path):
    if path.lower().endswith(".heic"):  # Pillow can't read HEIC; macOS sips can
        tmp = os.path.join(OUT, "_tmp.jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", path, "--out", tmp], capture_output=True, check=True)
        img = Image.open(tmp).convert("RGB")
        os.remove(tmp)
        return img
    return Image.open(path).convert("RGB")


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def fit(img, long_side=1800):
    if max(img.size) > long_side:
        img = img.copy()
        img.thumbnail((long_side, long_side), Image.LANCZOS)
    return img


def overlay(img, draw_fn):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer), img.size)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def wm_tiled(img):
    w, h = img.size
    size = max(24, w // 14)
    tile = Image.new("RGBA", (size * 8, size * 3), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.text((size // 2, size), "shutterstock", font=font(size), fill=(255, 255, 255, 110),
           stroke_width=max(1, size // 18), stroke_fill=(0, 0, 0, 60))
    tile = tile.rotate(28, expand=True)
    layer = Image.new("RGBA", (w + tile.width, h + tile.height), (0, 0, 0, 0))
    for y in range(0, h, int(tile.height * 0.7)):
        for x in range(0, w, int(tile.width * 0.8)):
            layer.alpha_composite(tile, (x, y))
    layer = layer.crop((0, 0, w, h))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def wm_center(img):
    def draw(d, size):
        w, h = size
        f = font(max(30, w // 9))
        text = "Getty Images"
        tw = d.textlength(text, font=f)
        d.text(((w - tw) / 2, h * 0.42), text, font=f, fill=(255, 255, 255, 130), stroke_width=2, stroke_fill=(0, 0, 0, 70))
        bar = max(22, h // 26)
        d.rectangle([0, h - bar, w, h], fill=(0, 0, 0, 170))
        d.text((10, h - bar + bar * 0.18), "Credit: Getty Images / Stock photo ID 128734921", font=font(int(bar * 0.6)), fill=(255, 255, 255, 230))
    return overlay(img, draw)


def wm_corner(img):
    def draw(d, size):
        w, h = size
        f = font(max(16, w // 32))
        text = "iStock by Getty Images"
        tw = d.textlength(text, font=f)
        d.text((w - tw - w * 0.03, h - h * 0.06), text, font=f, fill=(255, 255, 255, 200), stroke_width=1, stroke_fill=(0, 0, 0, 120))
        big = font(max(40, w // 6))
        tw = d.textlength("iStock", font=big)
        d.text(((w - tw) / 2, h * 0.4), "iStock", font=big, fill=(255, 255, 255, 80))
    return overlay(img, draw)


def variants(img):
    w, h = img.size
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(0)
    scale = 180 / min(w, h)
    small = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    bw, bh = w, int(w / 3.5)
    yield "blur", img.filter(ImageFilter.GaussianBlur(5 * max(w, h) / 1080))
    yield "tiny", small
    yield "dark", Image.fromarray(np.clip(arr * 0.16, 0, 255).astype(np.uint8))
    yield "bright", Image.fromarray(np.clip(arr * 2.4 + 70, 0, 255).astype(np.uint8))
    yield "noise", Image.fromarray(np.clip(arr + rng.normal(0, 32, arr.shape), 0, 255).astype(np.uint8))
    yield "wm_tiled", wm_tiled(img)
    yield "wm_center", wm_center(img)
    yield "wm_corner", wm_corner(img)
    if bh < h and bh >= 400:  # a banner cut from a small image is legitimately low-res
        top = (h - bh) // 2
        yield "banner", img.crop((0, top, bw, top + bh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", help="folder of clean source images (not committed)")
    ap.add_argument("--realworld", help="folder of real ads to score as-is (no expected verdict)")
    args = ap.parse_args()
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    rows = []

    def save(img, name, category, source):
        path = os.path.join(OUT, name)
        img.save(path, quality=92)
        rows.append({"file": os.path.relpath(path, EVAL_DIR), "category": category, "source": source})

    sources = sorted(p for p in glob.glob(os.path.join(args.sources or "", "*")) if p.lower().endswith(EXTS)) if args.sources else []
    apple = sorted(glob.glob(os.path.join(APPLE, "*", "*.heic")))[::4] if os.path.isdir(APPLE) else []
    for path in sources + apple:
        base = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")[:40]
        img = fit(load(path))
        if min(img.size) < 700:
            save(img, f"{base}__clean.jpg", "clean_lowres", path)
            continue
        save(img, f"{base}__clean.jpg", "clean", path)
        for kind, v in variants(img):
            save(v, f"{base}__{kind}.jpg", kind, path)
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "ai_generated", "*"))):
        if path.lower().endswith(EXTS):
            shutil.copy(path, os.path.join(OUT, "ai__" + os.path.basename(path)))
            rows.append({"file": os.path.join("images", "ai__" + os.path.basename(path)), "category": "ai", "source": path})

    for path in sorted(glob.glob(os.path.join(args.realworld or "", "*"))) if args.realworld else []:
        if path.lower().endswith(EXTS):
            name = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")[:40]
            save(fit(load(path)), f"real__{name}.jpg", "realworld", path)

    with open(os.path.join(EVAL_DIR, "image_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "category", "source"])
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"{len(rows)} images:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
