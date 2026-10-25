"""
03_web_app.py
-------------
Sentiment Studio web app: a FastAPI backend serving the site in web/.

    python 03_web_app.py                 # then open http://127.0.0.1:8000
    python 03_web_app.py --port 8080     # another port

Uses your fine-tuned model from models/best_model if it exists,
otherwise falls back to the public cardiffnlp checkpoint automatically.
Heavy models (sentiment, CLIP, Stable Diffusion, OCR) load lazily on first
use and stay in memory; the sentiment model and CLIP warm up in the
background at startup so the first request is fast.

Pages (see README.md for the full breakdown):
  Analyze       -> Ad Copy, Image Analysis, Bulk Analysis, Website Audit
  Create & Edit -> Generative Editing, Precise Editing, Canvas Editor
"""

import argparse
import base64
import importlib.util
import io
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

import copy_analysis as ca
from config import BASE_DIR, IMAGE_ENGAGEMENT, MODEL_DIR

WEB_DIR = os.path.join(BASE_DIR, "web")
MAX_UPLOAD_MB = 25


def _load_module_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE_DIR, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Module names starting with a digit can't be imported with a normal
# `import` statement - load them by path instead.
_model_training = _load_module_by_path("model_training", "02_model_training.py")
_image_analysis = _load_module_by_path("image_analysis", "04_image_analysis.py")
_generative_editing = _load_module_by_path("generative_editing", "05_generative_editing.py")
_image_editing = _load_module_by_path("image_editing", "06_image_editing.py")

OPTIONAL_DEPS = {
    "rembg": "Background removal (Precise Editing)",
    "easyocr": "Text detection (Precise Editing)",
    "diffusers": "Stable Diffusion (Generative Editing)",
    "requests": "Website Audit (fetching pages)",
    "bs4": "Website Audit (HTML parsing)",
}


class FriendlyError(Exception):
    """An error with a message meant for the user (shown as-is in the UI)."""

    def __init__(self, title: str, hint: str = "", status: int = 400):
        super().__init__(title)
        self.title, self.hint, self.status = title, hint, status


# ── Lazily loaded, shared heavy models ─────────────────────────────────
def _load_clip():
    try:
        return _image_analysis.ClipScorer()
    except Exception:  # noqa: BLE001 - scoring falls back to technical checks only
        return None


def _load_image_model():
    if not os.path.exists(IMAGE_ENGAGEMENT["model_path"]):
        return None
    try:
        return _load_module_by_path("image_model", "10_image_model_training.py").ImageEngagementPredictor()
    except Exception:  # noqa: BLE001
        return None


_LOADERS = {
    "sentiment": lambda: _model_training.SentimentPredictor(MODEL_DIR),
    "clip": _load_clip,
    "sd": lambda: _generative_editing.StableDiffusionEditor(),
    "image_model": _load_image_model,
    "website": lambda: _load_module_by_path("website_analysis", "09_website_analysis.py"),
}
_models: dict = {}
_locks = {name: threading.Lock() for name in _LOADERS}


def _cached_loader(name: str):
    """Loads each model once, even if several requests ask at the same time.
    A failed load isn't cached, so it's retried on the next request."""
    if name not in _models:
        with _locks[name]:
            if name not in _models:
                _models[name] = _LOADERS[name]()
    return _models[name]


def predictor():
    try:
        return _cached_loader("sentiment")
    except Exception as e:  # noqa: BLE001
        raise FriendlyError("Couldn't load the sentiment model.",
                            "Check your internet connection (the first run downloads a model) and that "
                            "torch/transformers are installed.", 503) from e


def clip_scorer():
    return _cached_loader("clip")


def _warm_up():
    for name in ("sentiment", "clip"):
        try:
            _cached_loader(name)
        except Exception:  # noqa: BLE001 - surfaced later, on first real request
            pass


# ── Helpers ──────────────────────────────────────────────────────────────
async def read_image(upload: UploadFile) -> Image.Image:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise FriendlyError(f"That image is larger than {MAX_UPLOAD_MB} MB.", "Please upload a smaller file.", 413)
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as e:
        raise FriendlyError("That file doesn't look like a valid image.", "Please upload a JPG, PNG or WebP.") from e
    return image.convert("RGB")


def to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def rgb_from_hex(hex_color: str) -> tuple:
    h = (hex_color or "#000000").lstrip("#")
    if len(h) != 6:
        raise FriendlyError("Invalid colour.", "Use a hex colour like #FF8800.")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def image_report(image: Image.Image) -> dict:
    """Quality score from 04 (+ the trained engagement model, if one exists)."""
    result = _image_analysis.score_image(image, clip_scorer=clip_scorer())
    result["features"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result["features"].items()}
    model = _cached_loader("image_model")
    result["trained_model"] = None
    if model is not None:
        try:
            p = model.predict(image)
            m = p["metrics"]
            result["trained_model"] = {"probability_top_third": p["probability_top_third"],
                                       "auc": m["auc"], "auc_ci": m["auc_ci"]}
        except Exception:  # noqa: BLE001
            pass
    return result


def require_dep(module: str, install: str):
    if importlib.util.find_spec(module) is None:
        raise FriendlyError(f"This feature needs the optional `{module}` package.",
                            f"Install it with `pip install {install}` and restart the app.", 501)


# ── App ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="Sentiment Studio", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)


@app.middleware("http")
async def no_cache_site_files(request, call_next):
    # the site is served from local files that change when the project is
    # updated - always revalidate so the browser never shows a stale UI
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.exception_handler(FriendlyError)
async def friendly_error_handler(_, exc: FriendlyError):
    return JSONResponse(status_code=exc.status, content={"error": exc.title, "hint": exc.hint})


@app.exception_handler(Exception)
async def unexpected_error_handler(_, exc: Exception):
    return JSONResponse(status_code=500, content={
        "error": "Something went wrong.", "hint": "Try again, or check the server console for details.",
        "detail": f"{type(exc).__name__}: {exc}"})


@app.get("/api/status")
def status():
    try:
        cuda = __import__("torch").cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda = False
    sentiment = _models.get("sentiment")
    using_fallback = sentiment.using_fallback if sentiment is not None else None
    return {
        "sentiment_model": {"ready": using_fallback is not None, "using_fallback": using_fallback,
                            "fine_tuned_available": os.path.isdir(MODEL_DIR)},
        "image_model_available": os.path.exists(IMAGE_ENGAGEMENT["model_path"]),
        "cuda": cuda,
        "optional_deps": [{"module": m, "description": d, "available": importlib.util.find_spec(m) is not None}
                          for m, d in OPTIONAL_DEPS.items()],
        "style_presets": list(_image_analysis.STYLE_PRESETS),
    }


# ── Ad copy ──────────────────────────────────────────────────────────────
class CopyRequest(BaseModel):
    text: str
    offer_details: str = ""


@app.post("/api/copy/analyze")
def analyze_copy(req: CopyRequest):
    text = req.text.strip()
    if not text:
        raise FriendlyError("Enter some ad copy to analyze.")
    result = predictor().predict([text])[0]
    features = ca.analyze_text_features(text)
    eng = ca.predict_engagement(features, result)
    refinement = ca.refine_low_engagement_copy(text, result["label"], eng["engagement_score"], req.offer_details)
    return {
        "sentiment": result,
        "explanation": ca.explain_sentiment(result, features),
        "features": features,
        "engagement": eng,
        "rewrite": {**refinement,
                    "grounded": not ca.find_unsupported_claims(refinement["refined_text"], text, req.offer_details)},
        "recommendation": ca.recommendation(result["label"]),
    }


# ── Image analysis ───────────────────────────────────────────────────────
@app.post("/api/image/analyze")
async def analyze_image(image: UploadFile = File(...)):
    img = await read_image(image)
    return image_report(img)


@app.post("/api/image/style")
async def style_image(image: UploadFile = File(...), preset: str = Form(...)):
    if preset not in _image_analysis.STYLE_PRESETS:
        raise FriendlyError(f"Unknown style preset: {preset}")
    img = await read_image(image)
    styled = _image_analysis.STYLE_PRESETS[preset](img)
    before, after = image_report(img), image_report(styled)
    return {"image": to_data_url(styled), "before": before["score"], "after": after["score"],
            "after_label": after["label"], "after_report": after}


# ── Bulk analysis (background job with progress) ────────────────────────
_jobs: dict = {}


def _run_bulk(job_id: str, df: pd.DataFrame, text_col: str, rewrite: bool):
    job = _jobs[job_id]
    try:
        texts = df[text_col].fillna("").astype(str).tolist()
        offer_col = next((c for c in df.columns if c.lower() in {"offer", "offer_details"}), None)
        offers = df[offer_col].fillna("").astype(str).tolist() if offer_col else [""] * len(texts)
        model = predictor()
        results = []
        for i in range(0, len(texts), 64):
            results.extend(model.predict(texts[i:i + 64]))
            job["progress"] = 0.5 * len(results) / len(texts) if rewrite else len(results) / len(texts)

        feats = [ca.analyze_text_features(t) for t in texts]
        engs = [ca.predict_engagement(f, r) for r, f in zip(results, feats)]
        out = df.copy()
        out["sentiment"] = [r["label"] for r in results]
        out["confidence"] = [r["confidence"] for r in results]
        for k, col in (("positive", "score_pos"), ("neutral", "score_neu"), ("negative", "score_neg")):
            out[col] = [r["scores"].get(k, 0) for r in results]
        out["sentiment_reason"] = [ca.explain_sentiment(r, f) for r, f in zip(results, feats)]
        out["heuristic_engagement_score"] = [round(e["engagement_score"], 1) for e in engs]
        out["engagement_label"] = [e["engagement_label"] for e in engs]
        out["engagement_reason"] = [" | ".join(e["reasons"]) for e in engs]

        if rewrite:
            refined = []
            for n, (t, r, e, o) in enumerate(zip(texts, results, engs, offers), 1):
                refined.append(ca.refine_low_engagement_copy(t, r["label"], e["engagement_score"], o))
                job["progress"] = 0.5 + 0.5 * n / len(texts)
            out["refined_text"] = [x["refined_text"] for x in refined]
            out["refined_heuristic_engagement_score"] = [round(x["after_score"], 1) for x in refined]
            out["heuristic_score_change"] = [round(x["score_change"], 1) for x in refined]
            out["rewrites_rejected_for_invented_claims"] = [len(x["rejected_candidates"]) for x in refined]

        counts = out["sentiment"].value_counts()
        job["result"] = {
            "text_column": text_col,
            "rows": len(out),
            "counts": {k: int(counts.get(k, 0)) for k in ("positive", "neutral", "negative")},
            "engagement_counts": {k: int((out["engagement_label"] == k).sum()) for k in ("high", "medium", "low")},
            "confidence": out[["sentiment", "confidence"]].to_dict(orient="records"),
            "preview": out.head(200).fillna("").to_dict(orient="records"),
            "columns": list(out.columns),
            "csv": out.to_csv(index=False),
        }
        job["status"], job["progress"] = "done", 1.0
    except FriendlyError as e:
        job.update(status="error", error=e.title, hint=e.hint)
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error="Bulk analysis failed.", hint=f"{type(e).__name__}: {e}")


@app.post("/api/bulk")
async def start_bulk(file: UploadFile = File(...), rewrite: bool = Form(True)):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e:  # noqa: BLE001
        raise FriendlyError("Couldn't read that CSV.", "Check the file is a valid, comma-separated CSV.") from e
    if df.empty or len(df.columns) == 0:
        raise FriendlyError("That CSV looks empty - nothing to analyze.")
    text_col = next((c for c in df.columns
                     if any(k in str(c).lower() for k in ["text", "content", "caption", "copy", "post"])), df.columns[0])
    # drop finished jobs older than an hour so memory doesn't grow forever
    for jid in [j for j, v in _jobs.items() if time.time() - v["created"] > 3600]:
        _jobs.pop(jid, None)
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "progress": 0.0, "created": time.time(), "rows": len(df), "text_column": text_col}
    threading.Thread(target=_run_bulk, args=(job_id, df, text_col, rewrite), daemon=True).start()
    return {"job_id": job_id, "rows": len(df), "text_column": text_col}


@app.get("/api/bulk/{job_id}")
def bulk_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise FriendlyError("That analysis job no longer exists.", "Start it again.", 404)
    return {k: v for k, v in job.items() if k != "created"}


# ── Website audit ────────────────────────────────────────────────────────
class WebsiteRequest(BaseModel):
    url: str
    pagespeed: bool = True
    sentiment: bool = True


@lru_cache(maxsize=64)
def _website_cached(url: str, pagespeed: bool, sentiment: bool, hour: int) -> dict:
    website = _cached_loader("website")
    return website.analyze_url(url, run_pagespeed=pagespeed, run_sentiment=sentiment,
                               predictor=predictor() if sentiment else None)


@app.post("/api/website")
def website_audit(req: WebsiteRequest):
    require_dep("requests", "requests")
    require_dep("bs4", "beautifulsoup4")
    if not req.url.strip():
        raise FriendlyError("Enter a page URL to audit.")
    website = _cached_loader("website")
    try:
        # results are cached for up to an hour per URL + options
        return _website_cached(req.url.strip(), req.pagespeed, req.sentiment, int(time.time() // 3600))
    except website.WebsiteAnalysisError as e:
        raise FriendlyError("Couldn't analyse that page.", str(e)) from e


# ── Generative editing ───────────────────────────────────────────────────
@app.get("/api/generate/estimate")
def generate_estimate(steps: int = 25):
    try:
        cuda = __import__("torch").cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda = False
    return {"estimate": _generative_editing.estimate_time_warning("cuda" if cuda else "cpu", steps)}


@app.post("/api/generate")
async def generate(image: UploadFile = File(...), prompt: str = Form(...), strength: float = Form(0.55),
                   steps: int = Form(25), guidance: float = Form(7.5)):
    require_dep("diffusers", "diffusers accelerate")
    if not prompt.strip():
        raise FriendlyError("Describe the new style or scene first.")
    img = await read_image(image)
    try:
        editor = _cached_loader("sd")
        generated = editor.generate(img, prompt=prompt, strength=strength, guidance_scale=guidance,
                                    num_inference_steps=steps)
    except ImportError as e:
        raise FriendlyError("Stable Diffusion isn't available.",
                            "Install the optional dependencies with `pip install diffusers accelerate`.", 501) from e
    except Exception as e:  # noqa: BLE001
        raise FriendlyError("Image generation failed.",
                            "This can happen with no internet connection (first-time model download), low "
                            "memory, or an unsupported prompt. Try a shorter prompt or fewer steps.", 500) from e
    return {"image": to_data_url(generated), "before": image_report(img)["score"],
            "after": image_report(generated)["score"]}


# ── Precise editing ──────────────────────────────────────────────────────
@app.post("/api/precise/background")
async def swap_background(image: UploadFile = File(...), color: str = Form("#FFFFFF")):
    require_dep("rembg", "rembg")
    img = await read_image(image)
    try:
        swapped = _image_editing.swap_background_to_color(img, rgb_from_hex(color))
    except ImportError as e:
        raise FriendlyError("Background removal isn't available.", "Install it with `pip install rembg`.", 501) from e
    return {"image": to_data_url(swapped), "before": image_report(img)["score"],
            "after": image_report(swapped)["score"]}


@app.post("/api/precise/detect-text")
async def detect_text(image: UploadFile = File(...)):
    require_dep("easyocr", "easyocr")
    img = await read_image(image)
    regions = _image_editing.detect_text_regions(img)
    return {"regions": [{**r, "bbox": list(r["bbox"])} for r in regions], "width": img.width, "height": img.height}


@app.post("/api/precise/replace-text")
async def replace_text(image: UploadFile = File(...), bbox: str = Form(...), text: str = Form(...),
                       color: str = Form("#000000")):
    img = await read_image(image)
    try:
        box = tuple(int(float(v)) for v in bbox.split(","))
        assert len(box) == 4
    except Exception as e:  # noqa: BLE001
        raise FriendlyError("Invalid text region.") from e
    edited = _image_editing.replace_text(img, box, text, text_color=rgb_from_hex(color))
    return {"image": to_data_url(edited), "before": image_report(img)["score"],
            "after": image_report(edited)["score"]}


# ── Static site ──────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main():
    import uvicorn

    ap = argparse.ArgumentParser(description="Run the Sentiment Studio web app.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = ap.parse_args()
    print(f"\n  Sentiment Studio → http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
