"""
03_dashboard.py
-----------------
Streamlit dashboard: sentiment analysis, heuristic engagement scoring,
and image/creative editing tools for marketing content.

    python -m streamlit run 03_dashboard.py --server.fileWatcherType none

Uses your fine-tuned model from models/best_model if it exists,
otherwise falls back to the public cardiffnlp checkpoint automatically.

Pages, grouped by what they do (see README.md for the full breakdown):
  Analyze       -> Ad Copy, Image Analysis, Bulk Analysis
  Create & Edit -> Generative Editing, Precise Editing, Canvas Editor
"""

import importlib.util
import io
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import MODEL_DIR

st.set_page_config(page_title="Sentiment Studio", page_icon="◆", layout="wide")


def _load_module_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Module names starting with a digit can't be imported with a normal
# `import` statement - load them by path instead. Wrapped in try/except
# so a broken/partial dependency install shows a friendly message
# instead of crashing the whole app with a raw traceback.
try:
    _model_training = _load_module_by_path("model_training", "02_model_training.py")
    SentimentPredictor = _model_training.SentimentPredictor
    _image_analysis = _load_module_by_path("image_analysis", "04_image_analysis.py")
    _generative_editing = _load_module_by_path("generative_editing", "05_generative_editing.py")
    _image_editing = _load_module_by_path("image_editing", "06_image_editing.py")
    _module_load_error = None
except Exception as e:
    _module_load_error = e

if _module_load_error is not None:
    st.error(
        "**Sentiment Studio couldn't start.**  \n"
        "One of the core modules failed to load - this usually means a required "
        "package (`torch`, `transformers`, ...) is missing or broken. "
        "Run `pip install -r Requirements.txt` and restart the app."
    )
    with st.expander("Technical details"):
        st.exception(_module_load_error)
    st.stop()

# ── Clean, restrained SaaS-style theme ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide the decorative menu/footer/deploy button, but keep the header's
   sidebar expand/collapse control working - hiding the whole header
   (as a naive `visibility:hidden` on header[data-testid="stHeader"]
   would) hides that control too, permanently trapping the sidebar
   closed with no way to reopen it. */
#MainMenu, footer, [data-testid="stAppDeployButton"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.stApp { background-color: #F7F8FB; }

h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif !important; color: #14182B !important; font-weight: 700 !important; }

.eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7280;
    margin: 0 0 0.5rem 0;
}

[data-testid="stStatusWidget"], .stSpinner, [data-testid="stSpinner"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
}
[data-testid="stStatusWidget"] *, .stSpinner *, [data-testid="stSpinner"] * {
    color: #14182B !important;
}

[data-testid="stCaptionContainer"], .stCaption, small {
    color: #6B7280 !important;
}
p, span, label, div { color: inherit; }
.stMarkdown p { color: #1F2430; }
[data-testid="stImageCaption"] { color: #374151 !important; font-weight: 500 !important; }

.flat-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}

.badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}

.heuristic-tag {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #7C6A00;
    background: #FEF9E7;
    border: 1px solid #F3E8A6;
    border-radius: 6px;
    padding: 0.12rem 0.5rem;
    margin-left: 0.5rem;
    vertical-align: middle;
}

.stTextArea textarea {
    border-radius: 10px !important;
    border: 1px solid #D8DCE3 !important;
    background-color: #FFFFFF !important;
    color: #1F2430 !important;
    caret-color: #1F2430 !important;
    font-family: 'Inter', sans-serif;
    padding: 0.85rem !important;
}

.stTextInput input {
    border-radius: 10px !important;
    border: 1px solid #D8DCE3 !important;
    background-color: #FFFFFF !important;
    color: #1F2430 !important;
    caret-color: #1F2430 !important;
}
.stTextInput label p, .stToggle label p, [data-testid="stWidgetLabel"] p { color: #374151 !important; }

.note {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 3px solid #CBD5E1;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 0.2rem 0 1rem 0;
    color: #475569;
    font-size: 0.86rem;
    line-height: 1.5;
}
.note b { color: #1F2430; }
.note code { font-size: 0.8rem; background: #F1F5F9; padding: 0.05rem 0.3rem; border-radius: 4px; }

.stButton button, .stDownloadButton button {
    border-radius: 9px !important;
    background-color: #4F46E5 !important;
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.55rem 1.3rem !important;
}
.stButton button:hover, .stDownloadButton button:hover { background-color: #4338CA !important; }
.stButton button[kind="secondary"] { background-color: #FFFFFF !important; color: #374151 !important; border: 1px solid #D8DCE3 !important; }
.stButton button[kind="secondary"]:hover { background-color: #F3F4F6 !important; }

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
}
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #6B7280 !important; text-transform: uppercase; letter-spacing: 0.05em; }

.stDataFrame, [data-testid="stFileUploader"] {
    border-radius: 10px !important;
    border: 1px solid #E5E7EB !important;
}

.stTabs [data-baseweb="tab"] { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 600; font-size: 0.9rem; }

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] * { color: #14182B !important; }
[data-testid="stSidebar"] hr { border-color: #E5E7EB; margin: 1.1rem 0; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: 0.15rem 0; }

div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.25rem; }

.stProgress > div > div > div > div { background-color: #4F46E5 !important; }
</style>
""", unsafe_allow_html=True)

COLORS = {"positive": "#16A34A", "neutral": "#B45309", "negative": "#DC2626"}
BADGE_BG = {"positive": "#DCFCE7", "neutral": "#FEF3C7", "negative": "#FEE2E2"}
TIER_COLORS = {"high": "#16A34A", "medium": "#B45309", "low": "#DC2626"}
TIER_BG = {"high": "#DCFCE7", "medium": "#FEF3C7", "low": "#FEE2E2"}
QUALITY_LABELS = {"high": "GOOD", "medium": "FAIR", "low": "NEEDS WORK"}

POSITIVE_WORDS = {
    "amazing", "best", "great", "boost", "improve", "win", "easy", "powerful",
    "fast", "free", "exclusive", "new", "save", "trusted", "proven", "love",
}
NEGATIVE_WORDS = {
    "bad", "problem", "issue", "delay", "fail", "broken", "worst", "slow",
    "error", "late", "down", "unhappy", "hard", "difficult",
}
CTA_WORDS = {"try", "start", "get", "shop", "buy", "join", "download", "learn", "book", "claim"}
URGENCY_WORDS = {"now", "today", "limited", "hurry", "last", "deadline", "only"}

OPTIONAL_DEPS = {
    "rembg": "Background removal (Precise Editing)",
    "easyocr": "Text detection (Precise Editing)",
    "diffusers": "Stable Diffusion (Generative Editing)",
    "streamlit_drawable_canvas": "Canvas Editor",
}


# ── Small reusable UI components (avoid repeating raw HTML blocks) ────
def eyebrow(text: str) -> None:
    st.markdown(f'<p class="eyebrow">{text}</p>', unsafe_allow_html=True)


def info_card(html_inner: str, style: str = "color:#334155; font-size:0.92rem;") -> None:
    st.markdown(f'<div class="flat-card" style="{style}">{html_inner}</div>', unsafe_allow_html=True)


def note(html_inner: str) -> None:
    """Low-key explanatory note (replaces the bright blue st.info boxes)."""
    st.markdown(f'<div class="note">{html_inner}</div>', unsafe_allow_html=True)


def reason_card(reasons: list) -> None:
    info_card("<br>".join(f"• {r}" for r in reasons))


def kpi_card(label: str, badge_text: str, badge_bg: str, badge_fg: str, subtext: str,
             footnote: str = "", heuristic: bool = False) -> None:
    tag = '<span class="heuristic-tag">Heuristic</span>' if heuristic else ""
    footnote_html = f'<p style="color:#94A3B8; margin:0.3rem 0 0; font-size:0.78rem;">{footnote}</p>' if footnote else ""
    st.markdown(f"""
    <div class="flat-card">
        <p class="eyebrow">{label}{tag}</p>
        <span class="badge" style="background:{badge_bg}; color:{badge_fg};">{badge_text}</span>
        <p style="color:#475569; margin:0.6rem 0 0; font-size:0.9rem;">{subtext}</p>
        {footnote_html}
    </div>
    """, unsafe_allow_html=True)


def friendly_error(title: str, hint: str, exc: Exception = None) -> None:
    st.error(f"**{title}**  \n{hint}")
    if exc is not None:
        debug = st.session_state.get("debug_mode", False)
        if debug:
            with st.expander("Technical details"):
                st.exception(exc)
        else:
            st.caption("Turn on \"Show technical details\" in the sidebar to see the full error.")


def visual_breakdown(sub_scores: dict, features: dict) -> None:
    rows = [
        ("Brightness", sub_scores["brightness"]),
        ("Contrast", sub_scores["contrast"]),
        ("Saturation", sub_scores["saturation"]),
        ("Sharpness", sub_scores["sharpness"]),
        ("Aspect ratio fit", sub_scores["aspect_ratio"]),
    ]
    for name, val in rows:
        st.caption(f"{name} · {val:.0f}/100")
        st.progress(min(1.0, max(0.0, val / 100.0)))
    st.caption(f"{features['width']}×{features['height']}px · ratio {features['aspect_ratio']:.2f}")


def quality_tier(score: float) -> str:
    return "high" if score >= 70 else "medium" if score >= 45 else "low"


def rgb_from_hex(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def image_download_button(image, label: str, filename: str) -> None:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    st.download_button(label, buf.getvalue(), filename, "image/png")


def reset_state_on_new_upload(uploaded_file, state_key: str, *keys_to_clear: str) -> None:
    """Uploaders keep old session_state (detected text regions, previous
    edit output...) around across re-runs. Without this, switching to a
    new image can show results computed from the PREVIOUS image."""
    if uploaded_file is None:
        return
    identity = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get(state_key) != identity:
        st.session_state[state_key] = identity
        for key in keys_to_clear:
            st.session_state.pop(key, None)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def open_uploaded_image(uploaded_file):
    from PIL import Image, UnidentifiedImageError
    try:
        return Image.open(uploaded_file)
    except UnidentifiedImageError:
        st.error("That file doesn't look like a valid image. Please upload a JPG or PNG.")
        return None
    except Exception as e:
        friendly_error("Couldn't open that image.", "Try a different file.", e)
        return None


# ── Feature analysis + explanations (sentiment/engagement logic, unchanged) ──
def analyze_text_features(text: str) -> dict:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    lower = cleaned.lower()
    tokens = re.findall(r"[a-zA-Z']+", lower)
    token_set = set(tokens)

    return {
        "cleaned_text": cleaned,
        "word_count": len(tokens),
        "char_count": len(cleaned),
        "has_exclamation": "!" in cleaned,
        "has_question": "?" in cleaned,
        "positive_hits": sorted(token_set & POSITIVE_WORDS),
        "negative_hits": sorted(token_set & NEGATIVE_WORDS),
        "cta_hits": sorted(token_set & CTA_WORDS),
        "urgency_hits": sorted(token_set & URGENCY_WORDS),
    }


def explain_sentiment(result: dict, features: dict) -> str:
    label = result["label"]
    conf = result["confidence"]
    scores = result.get("scores", {})
    pos, neu, neg = (float(scores.get(k, 0.0)) for k in ("positive", "neutral", "negative"))
    ranked = sorted([("positive", pos), ("neutral", neu), ("negative", neg)], key=lambda x: x[1], reverse=True)
    runner_up_label, runner_up_score = ranked[1]
    margin = conf - runner_up_score

    reasons = [
        f"Predicted `{label}` with {conf:.1%} confidence "
        f"(positive {pos:.1%}, neutral {neu:.1%}, negative {neg:.1%}).",
        f"It is `{label}` because this score is {margin:.1%} higher than `{runner_up_label}`.",
    ]
    if features["positive_hits"] and not features["negative_hits"]:
        reasons.append(f"Language leans positive via words like: {', '.join(features['positive_hits'][:4])}.")
    elif features["negative_hits"] and not features["positive_hits"]:
        reasons.append(f"Language leans negative via words like: {', '.join(features['negative_hits'][:4])}.")
    elif features["positive_hits"] and features["negative_hits"]:
        reasons.append("Mixed cues detected (positive and negative words both appear), which often pushes prediction toward neutral.")
    else:
        reasons.append("No strong polarity keywords were found, so the model relies more on overall tone.")

    if features["word_count"] < 6:
        reasons.append("Very short copy gives limited context, which commonly increases neutral predictions.")
    if features["has_question"] and label == "neutral":
        reasons.append("Question-style phrasing can be informative rather than emotional, supporting a neutral tone.")
    if features["has_exclamation"] and label != "negative":
        reasons.append("Exclamation adds energy, which may support positive/engaging interpretation.")
    return " ".join(reasons)


def predict_engagement(features: dict, result: dict) -> dict:
    """Rule-based (heuristic) engagement score, 0-100. Not a trained
    model - see README for why no validated text-engagement model exists
    yet. Always label this as "heuristic" in the UI."""
    score = 50.0
    reasons = []
    pos = result["scores"].get("positive", 0.0)
    neg = result["scores"].get("negative", 0.0)
    neu = result["scores"].get("neutral", 0.0)

    score += (pos - neg) * 35
    if features["cta_hits"]:
        score += 12
        reasons.append(f"Contains CTA words ({', '.join(features['cta_hits'][:3])}).")
    else:
        score -= 10
        reasons.append("Missing a clear call-to-action.")

    if features["urgency_hits"]:
        score += 8
        reasons.append(f"Includes urgency words ({', '.join(features['urgency_hits'][:3])}).")
    else:
        reasons.append("No urgency trigger detected.")

    wc = features["word_count"]
    if 8 <= wc <= 24:
        score += 10
        reasons.append("Length is in a strong engagement range (8-24 words).")
    elif wc < 8:
        score -= 12
        reasons.append("Copy is too short to communicate value clearly.")
    else:
        score -= 6
        reasons.append("Copy is long and may reduce attention.")

    if features["has_exclamation"]:
        score += 4
        reasons.append("Emotional punctuation can improve attention.")
    if neu > 0.55:
        score -= 8
        reasons.append("Neutral tone may feel less persuasive.")

    score = float(np.clip(score, 0, 100))
    label = "high" if score >= 70 else "medium" if score >= 45 else "low"
    return {"engagement_score": score, "engagement_label": label, "reasons": reasons}


# ── Copy refinement (local rewriter, with template fallback) ─────────
def sentiment_prior_from_label(label: str) -> dict:
    label = str(label or "").lower()
    if label == "positive":
        scores = {"positive": 0.75, "neutral": 0.20, "negative": 0.05}
    elif label == "negative":
        scores = {"positive": 0.20, "neutral": 0.20, "negative": 0.60}
    else:
        scores = {"positive": 0.30, "neutral": 0.55, "negative": 0.15}
    return {"label": label if label in {"positive", "neutral", "negative"} else "neutral", "scores": scores}


class _Seq2SeqRewriter:
    """Minimal drop-in for `transformers.pipeline("text2text-generation", ...)`.

    That task name was removed from the pipeline task registry in newer
    transformers releases (confirmed: raises KeyError on transformers>=5),
    so this calls AutoModelForSeq2SeqLM.generate() directly instead - that
    API is stable across transformers versions. Mimics the pipeline's
    call signature/output shape so the rest of the code doesn't change."""

    def __init__(self, model_name: str, device: int):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device
        if device >= 0:
            self.model = self.model.to(f"cuda:{device}")

    def __call__(self, prompt: str, max_new_tokens: int = 64, do_sample: bool = True,
                 temperature: float = 0.9, top_p: float = 0.95, num_return_sequences: int = 1):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        if self.device >= 0:
            inputs = {k: v.to(f"cuda:{self.device}") for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p, num_return_sequences=num_return_sequences,
        )
        return [{"generated_text": self.tokenizer.decode(o, skip_special_tokens=True)} for o in outputs]


@st.cache_resource(show_spinner=False)
def load_local_rewriter():
    try:
        import torch
        model_name = os.getenv("LOCAL_REWRITE_MODEL", "google/flan-t5-base")
        device = 0 if torch.cuda.is_available() else -1
        return _Seq2SeqRewriter(model_name, device)
    except Exception:
        return None


# ── Grounding / anti-hallucination checks for the rewriter ──────────
# A rewrite must never invent facts the advertiser didn't give us: no new
# deadlines ("today only"), prices, discounts, percentages, guarantees or
# superlative claims. Anything like that is only allowed if it already
# appears in the original copy or in the optional "offer details" the
# user typed in. Otherwise the tool would be producing false advertising.
CLAIM_PHRASES = [
    "today only", "only today", "limited time", "limited-time", "limited offer",
    "ends tonight", "ends today", "ends soon", "last chance", "while stocks last",
    "while supplies last", "act now", "hurry", "this weekend only", "flash sale",
    "sale", "discount", "off", "free", "free shipping", "guarantee", "guaranteed",
    "money back", "money-back", "proven", "clinically", "certified", "award",
    "award-winning", "best-selling", "bestselling", "#1", "number one", "no.1",
    "exclusive", "bonus", "deal", "coupon", "promo",
]
_NUMBER_RE = re.compile(r"[$₹€£]?\d[\d,.:]*\s?(%|percent|x|am|pm|hrs?|hours?|days?|mins?)?", re.I)
_DATE_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "tonight", "tomorrow", "weekend", "midnight",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def extract_claims(text: str) -> set:
    """Every factual 'claim token' in a piece of copy: numbers/prices/
    percentages, date or deadline words, and promotional claim phrases."""
    t = _norm(text)
    claims = {m.group(0).strip().lower() for m in _NUMBER_RE.finditer(t) if m.group(0).strip()}
    tokens = set(re.findall(r"[a-z#0-9.'-]+", t))
    claims |= tokens & _DATE_WORDS
    for phrase in CLAIM_PHRASES:
        if " " in phrase or "-" in phrase or "#" in phrase or "." in phrase:
            if phrase in t:
                claims.add(phrase)
        elif phrase in tokens:
            claims.add(phrase)
    return claims


def find_unsupported_claims(candidate: str, source_text: str, offer_details: str = "") -> list:
    """Claims present in `candidate` that are NOT backed by the original
    copy or the user-supplied offer details. Non-empty = hallucination."""
    allowed_text = _norm(f"{source_text} {offer_details}")
    allowed = extract_claims(allowed_text)
    unsupported = []
    for claim in sorted(extract_claims(candidate)):
        if claim in allowed or claim in allowed_text:
            continue
        unsupported.append(claim)
    return unsupported


def safe_template_rewrite(base_text: str, offer_details: str = "") -> str:
    """Deterministic fallback rewrite. It only restructures the user's own
    words and adds a neutral call-to-action. Urgency or offer language
    appears ONLY if the user supplied it in `offer_details`."""
    core = re.sub(r"\s+", " ", str(base_text or "")).strip().strip(".!?")
    if not core:
        return ""
    core = core[0].upper() + core[1:]

    lowered = core.lower()
    has_cta = any(re.search(rf"\b{w}\b", lowered) for w in CTA_WORDS)
    seed = sum(ord(c) for c in core) % 4
    neutral_ctas = ["Learn more", "Try it", "Find out more", "See how it works"]

    parts = [f"{core}."]
    offer = re.sub(r"\s+", " ", str(offer_details or "")).strip().strip(".!?")
    if offer:
        parts.append(f"{offer[0].upper() + offer[1:]}.")
    if not has_cta:
        parts.append(f"{neutral_ctas[seed]}!")
    return " ".join(parts)


def content_relevance_bonus(source_text: str, candidate_text: str) -> float:
    source_tokens = set(re.findall(r"[A-Za-z0-9']+", source_text.lower()))
    candidate_tokens = set(re.findall(r"[A-Za-z0-9']+", candidate_text.lower()))
    if not source_tokens:
        return 0.0
    return (len(source_tokens & candidate_tokens) / max(len(source_tokens), 1)) * 8.0


def build_candidate_pool(raw_candidates, base_text: str):
    """Light cleanup only. Earlier versions force-appended 'today only',
    'Try now!' and 'Feel the difference!' to every candidate, which both
    invented offers and gamed the heuristic score. That is removed: a
    candidate is kept as the model wrote it, then grounding-checked."""
    pool, seen = [], set()
    for text in raw_candidates:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip().strip('"').strip("'").replace("\n", " ").strip()
        cleaned = re.sub(r"[^\w\s,!?.'%$₹€£-]", "", cleaned)
        if not cleaned:
            continue
        cleaned = cleaned[0].upper() + cleaned[1:]
        words = cleaned.split()
        if len(words) > 25:
            cleaned = " ".join(words[:25]).rstrip(",;:-") + "."
        key = cleaned.lower()
        if key and key not in seen and key != base_text.lower() and len(words) >= 3:
            seen.add(key)
            pool.append(cleaned)
    return pool


def _refinement_result(base, text, before, prior, rejected, method):
    after = predict_engagement(analyze_text_features(text), prior)["engagement_score"]
    return {
        "refined_text": text, "before_score": before, "after_score": after,
        "score_change": after - before, "rejected_candidates": rejected,
        "unsupported_claims": find_unsupported_claims(text, base),
        "method": method,
    }


@st.cache_data(show_spinner=False)
def refine_low_engagement_copy(text: str, sentiment_label: str, engagement_score: float,
                               offer_details: str = "") -> dict:
    base = re.sub(r"\s+", " ", str(text or "")).strip()
    offer_details = re.sub(r"\s+", " ", str(offer_details or "")).strip()
    prior = sentiment_prior_from_label(sentiment_label)
    if not base:
        return {"refined_text": "", "before_score": engagement_score, "after_score": engagement_score,
                "score_change": 0.0, "rejected_candidates": [], "unsupported_claims": [], "method": "none"}

    rewriter = load_local_rewriter()
    if rewriter is None:
        return _refinement_result(base, safe_template_rewrite(base, offer_details), engagement_score,
                                  prior, [], "template (no language model loaded)")

    facts = f"Facts you may use: {offer_details}\n" if offer_details else (
        "There is NO sale, discount, deadline or special offer. Do not mention one.\n")
    prompt = (
        "Rewrite this ad copy to be clearer and more persuasive, with a call-to-action.\n"
        "Do not add any prices, numbers, discounts, deadlines, guarantees or claims "
        "that are not in the original or the facts below. Max 20 words.\n"
        f"{facts}Original: {base}\nRewrite:"
    )
    try:
        generated = rewriter(prompt, max_new_tokens=64, do_sample=True, temperature=0.8,
                             top_p=0.95, num_return_sequences=8)
        candidates = build_candidate_pool([g.get("generated_text", "") for g in generated], base)
        candidates.append(safe_template_rewrite(base, offer_details))

        best_text, best_score, rejected = base, float("-inf"), []
        for candidate in candidates:
            unsupported = find_unsupported_claims(candidate, base, offer_details)
            if unsupported:
                rejected.append({"text": candidate, "unsupported": unsupported})
                continue
            score = predict_engagement(analyze_text_features(candidate), prior)["engagement_score"]
            score += content_relevance_bonus(base, candidate)
            if score > best_score:
                best_score, best_text = score, candidate
        return _refinement_result(base, best_text, engagement_score, prior, rejected,
                                  "language model + grounding check")
    except Exception:
        return _refinement_result(base, safe_template_rewrite(base, offer_details), engagement_score,
                                  prior, [], "template (language model failed)")


# ── Cached heavy-model loaders ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_predictor():
    return SentimentPredictor(MODEL_DIR)


@st.cache_resource(show_spinner="Loading CLIP model (first run only)...")
def load_clip_scorer():
    try:
        return _image_analysis.ClipScorer()
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading Stable Diffusion (first run downloads ~4-5GB)...")
def load_sd_editor():
    return _generative_editing.StableDiffusionEditor()


# ── Model loading (fails loudly but gracefully, not with a raw traceback) ──
try:
    predictor = load_predictor()
    predictor_error = None
except Exception as e:
    predictor = None
    predictor_error = e

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:baseline; justify-content:space-between; margin-bottom:1.2rem;">
    <div>
        <h1 style="margin:0; font-size:1.6rem;">Sentiment Studio</h1>
        <p style="margin:0.2rem 0 0; color:#6B7280; font-size:0.9rem;">AI-assisted marketing copy &amp; creative analysis</p>
    </div>
</div>
""", unsafe_allow_html=True)

if predictor_error is not None:
    friendly_error(
        "Couldn't load the sentiment model.",
        "The app can't run without it. Check your internet connection (first run "
        "downloads a model) or that `transformers`/`torch` are installed correctly.",
        predictor_error,
    )
    st.stop()


# ── Sidebar: grouped navigation + system status ────────────────────────
NAV = {
    "Analyze": ["Ad Copy", "Image Analysis", "Bulk Analysis", "Website Analysis"],
    "Create & Edit": ["Generative Editing", "Precise Editing", "Canvas Editor"],
}

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Plus Jakarta Sans\',sans-serif; font-weight:800; '
        'font-size:1.05rem; margin-bottom:1.2rem;">◆ Sentiment Studio</div>',
        unsafe_allow_html=True,
    )

    eyebrow("Section")
    section = st.radio("Section", list(NAV.keys()), label_visibility="collapsed", key="nav_section")

    eyebrow(section)
    mode = st.radio("Page", NAV[section], label_visibility="collapsed", key=f"nav_page_{section}")

    st.markdown("---")
    eyebrow("Settings")
    show_scores = st.toggle("Show all class scores", value=True)
    debug_mode = st.toggle("Show technical details on errors", value=False, key="debug_mode")

    st.markdown("---")
    eyebrow("System")
    status = "Custom fine-tuned model" if not predictor.using_fallback else "Public baseline model"
    st.caption(f"Sentiment model: {status}")
    if predictor.using_fallback:
        st.caption("No fine-tuned model in `models/best_model` yet, so the public "
                   "cardiffnlp/twitter-roberta-base-sentiment-latest model is used. "
                   "Run `python 02_model_training.py` to use your own.")
    with st.expander("Optional dependencies"):
        for pkg, desc in OPTIONAL_DEPS.items():
            available = importlib.util.find_spec(pkg) is not None
            st.caption(f"{'✓' if available else '○'} {desc} — {'available' if available else 'not installed'}")

# ══════════════════════════════════════════════════════════════════════
# Analyze > Ad Copy
# ══════════════════════════════════════════════════════════════════════
if mode == "Ad Copy":
    eyebrow("Analyze")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_text = st.text_area(
            "Ad or post copy",
            placeholder="e.g. Boost your productivity with our amazing AI tool — try free today!",
            height=120,
            label_visibility="collapsed",
        )
        offer_details = st.text_input(
            "Real offer details (optional)",
            placeholder="e.g. 20% off until Sunday, free shipping over ₹999",
            help="The rewrite may only mention prices, discounts, deadlines or guarantees that "
                 "appear in your copy or here. Leave blank if there is no offer.",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Analyze", use_container_width=True, type="primary")

    if run_btn and user_text.strip():
        with st.spinner("Analyzing..."):
            result = predictor.predict([user_text])[0]

        label, conf = result["label"], result["confidence"]
        color, bg = COLORS.get(label, "#4F46E5"), BADGE_BG.get(label, "#E0E7FF")
        features = analyze_text_features(user_text)
        sentiment_reason = explain_sentiment(result, features)
        eng = predict_engagement(features, result)
        eng_color, eng_bg = TIER_COLORS[eng["engagement_label"]], TIER_BG[eng["engagement_label"]]

        m1, m2 = st.columns(2)
        with m1:
            kpi_card("Sentiment", label.upper(), bg, color, f"Confidence {conf:.1%}")
        with m2:
            kpi_card(
                "Engagement", eng["engagement_label"].upper(), eng_bg, eng_color,
                f"Score {eng['engagement_score']:.0f} / 100", heuristic=True,
            )

        if show_scores:
            eyebrow("Score breakdown")
            scores = result["scores"]
            fig, ax = plt.subplots(figsize=(4, 2.2))
            bars = ax.bar(list(scores.keys()), list(scores.values()),
                           color=[COLORS[k] for k in scores.keys()], edgecolor="none", width=0.55)
            ax.set_ylim(0, 1.1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            ax.set_yticklabels([f"{t:.0%}" for t in ax.get_yticks()])
            for bar, v in zip(bars, scores.values()):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.1%}", ha="center", fontsize=8, color="#334155")
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            chart_col, _ = st.columns([1, 2])
            with chart_col:
                st.pyplot(fig, use_container_width=True)

        eyebrow("Why this result")
        info_card(sentiment_reason)

        eyebrow("Engagement factors")
        st.caption("Based on the heuristic scoring rules below (keyword/length checks), not a trained engagement model.")
        reason_card(eng["reasons"])

        refinement = refine_low_engagement_copy(user_text, label, eng["engagement_score"], offer_details)
        eyebrow("Suggested rewrite")
        st.code(refinement["refined_text"], language="text")
        st.caption(
            f"Heuristic score (not a measured result) — before {refinement['before_score']:.0f}/100 · "
            f"after {refinement['after_score']:.0f}/100 · "
            f"change {refinement['score_change']:+.0f} · method: {refinement['method']}"
        )
        st.caption(
            "✓ Grounding check passed: the rewrite adds no prices, discounts, deadlines or "
            "guarantees beyond your copy and offer details."
            if not find_unsupported_claims(refinement["refined_text"], user_text, offer_details)
            else "⚠ Rewrite contains claims not found in your input — review before using."
        )
        if refinement["rejected_candidates"]:
            with st.expander(f"{len(refinement['rejected_candidates'])} generated rewrite(s) rejected for invented claims"):
                for r in refinement["rejected_candidates"]:
                    st.markdown(f"- {r['text']}  \n  *unsupported:* `{', '.join(r['unsupported'])}`")

        eyebrow("Recommendation")
        if label == "negative":
            info_card("Negative tone detected. Consider reframing pain points as solutions, using positive action verbs, and adding a clear benefit statement.")
        elif label == "neutral":
            info_card("Neutral tone. To boost engagement, add emotional triggers, urgency words ('now', 'today', 'exclusive'), or a stronger call-to-action.")
        else:
            info_card("Positive tone — likely to resonate well with audiences. Pair with high-quality visuals for maximum impact.")

# ══════════════════════════════════════════════════════════════════════
# Analyze > Image Analysis
# ══════════════════════════════════════════════════════════════════════
elif mode == "Image Analysis":
    eyebrow("Image engagement")
    st.caption("Evaluate how visually engaging your creative is.")
    note(
        "This score is a <b>heuristic</b>, not a trained/validated model like the sentiment "
        "classifier. There's no real dataset pairing marketing images with actual engagement "
        "outcomes to train or validate against, so this combines a pretrained CLIP model's "
        "semantic judgement with measurable visual properties (brightness, contrast, "
        "saturation, sharpness, aspect ratio). Treat it as directional guidance, not a "
        "validated prediction."
    )

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="analysis_upload")

    if uploaded_img is not None:
        image = open_uploaded_image(uploaded_img)

    if uploaded_img is not None and image is not None:
        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Scoring image..."):
            clip_scorer = load_clip_scorer()
            result = _image_analysis.score_image(image, clip_scorer=clip_scorer)

        tier = result["label"]
        with col_result:
            kpi_card(
                "Engagement", tier.upper(), TIER_BG[tier], TIER_COLORS[tier],
                f"Score {result['score']:.0f} / 100",
                footnote="CLIP + visual features" if result["used_clip"] else "Visual features only (CLIP unavailable)",
                heuristic=True,
            )

        eyebrow("Visual breakdown")
        visual_breakdown(result["sub_scores"], result["features"])

        eyebrow("Why this score")
        reason_card(result["reasons"])

        eyebrow("Improve your creative")
        preset_name = st.selectbox("Style", list(_image_analysis.STYLE_PRESETS.keys()))
        refine_clicked = st.button("Apply style")

        if refine_clicked:
            preset_fn = _image_analysis.STYLE_PRESETS[preset_name]
            refined_img = preset_fn(image)
            refined_result = _image_analysis.score_image(refined_img, clip_scorer=clip_scorer)

            col_before, col_after = st.columns(2)
            with col_before:
                st.image(image, caption=f"Before ({result['score']:.0f}/100)", use_container_width=True)
            with col_after:
                st.image(refined_img, caption=f"After - {preset_name} ({refined_result['score']:.0f}/100)", use_container_width=True)

            uplift = refined_result["score"] - result["score"]
            st.caption(f"Heuristic score uplift: {uplift:+.0f}")
            if uplift <= 0:
                st.caption("This style didn't raise the heuristic score for this image - try a different preset, or it may already be close to optimal.")

            image_download_button(refined_img, "Download styled image", f"{preset_name.lower().replace(' ', '_').replace('/', '')}.png")

        st.caption("Need exact background swaps or text replacement? Use the **Precise Editing** page under Create & Edit.")

# ══════════════════════════════════════════════════════════════════════
# Analyze > Bulk Analysis
# ══════════════════════════════════════════════════════════════════════
elif mode == "Bulk Analysis":
    eyebrow("Bulk analysis")
    st.caption("Upload a CSV with a column named `text` containing your ad copies.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            friendly_error("Couldn't read that CSV.", "Check the file is a valid, comma-separated CSV.", e)
            st.stop()
        if df.empty or len(df.columns) == 0:
            st.warning("That CSV looks empty — nothing to analyze.")
            st.stop()

        st.caption(f"Loaded {len(df):,} rows.")
        text_col = next((c for c in df.columns if any(k in c.lower() for k in ["text", "content", "caption", "copy", "post"])), df.columns[0])
        st.caption(f"Using column: `{text_col}`")

        if st.button("Run analysis", type="primary"):
            texts = df[text_col].fillna("").astype(str).tolist()
            progress = st.progress(0, text="Analyzing...")
            results = []
            batch_size = 64
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                results.extend(predictor.predict(batch))
                progress.progress(min((i + batch_size) / len(texts), 1.0))

            df["sentiment"] = [r["label"] for r in results]
            df["confidence"] = [r["confidence"] for r in results]
            df["score_pos"] = [r["scores"].get("positive", 0) for r in results]
            df["score_neu"] = [r["scores"].get("neutral", 0) for r in results]
            df["score_neg"] = [r["scores"].get("negative", 0) for r in results]
            feature_rows = [analyze_text_features(t) for t in texts]
            df["sentiment_reason"] = [explain_sentiment(r, f) for r, f in zip(results, feature_rows)]
            engagement_rows = [predict_engagement(f, r) for r, f in zip(results, feature_rows)]
            offer_col = next((c for c in df.columns if c.lower() in {"offer", "offer_details"}), None)
            offers = df[offer_col].fillna("").astype(str).tolist() if offer_col else [""] * len(texts)
            refined_rows = [refine_low_engagement_copy(t, r["label"], e["engagement_score"], o)
                            for t, r, e, o in zip(texts, results, engagement_rows, offers)]
            df["heuristic_engagement_score"] = [e["engagement_score"] for e in engagement_rows]
            df["engagement_label"] = [e["engagement_label"] for e in engagement_rows]
            df["engagement_reason"] = [" | ".join(e["reasons"]) for e in engagement_rows]
            df["refined_text"] = [x["refined_text"] for x in refined_rows]
            df["refined_heuristic_engagement_score"] = [x["after_score"] for x in refined_rows]
            df["heuristic_score_change"] = [x["score_change"] for x in refined_rows]
            df["rewrites_rejected_for_invented_claims"] = [len(x["rejected_candidates"]) for x in refined_rows]
            progress.empty()

            counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", counts.get("positive", 0))
            col2.metric("Neutral", counts.get("neutral", 0))
            col3.metric("Negative", counts.get("negative", 0))

            fig1, ax1 = plt.subplots(figsize=(5, 5))
            nonzero = counts[counts > 0]
            ax1.pie(nonzero.values, labels=nonzero.index, autopct="%1.0f%%",
                    colors=[COLORS[k] for k in nonzero.index],
                    wedgeprops={"edgecolor": "#FFFFFF", "linewidth": 2})
            ax1.set_title("Sentiment distribution", fontsize=11, color="#334155")
            fig1.patch.set_alpha(0)
            st.pyplot(fig1, use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            for label in ["positive", "neutral", "negative"]:
                subset = df.loc[df["sentiment"] == label, "confidence"]
                if len(subset):
                    ax2.hist(subset, bins=20, alpha=0.7, label=label, color=COLORS[label])
            ax2.spines[["top", "right"]].set_visible(False)
            ax2.set_xlabel("Confidence")
            ax2.set_ylabel("Count")
            ax2.set_title("Confidence distribution by sentiment", fontsize=11, color="#334155")
            ax2.legend()
            fig2.patch.set_alpha(0)
            st.pyplot(fig2, use_container_width=True)

            st.dataframe(df[[text_col, "sentiment", "confidence"]].head(50), use_container_width=True)
            eyebrow("Heuristic engagement preview")
            st.dataframe(df[[text_col, "engagement_label", "heuristic_engagement_score", "refined_heuristic_engagement_score", "heuristic_score_change"]].head(50), use_container_width=True)
            eyebrow("Refined text")
            st.dataframe(
                df[[text_col, "refined_text"]].head(50),
                column_config={
                    text_col: st.column_config.TextColumn("Original text", width=420),
                    "refined_text": st.column_config.TextColumn("Refined text", width=600),
                },
            )
            st.download_button("Download results CSV", df.to_csv(index=False).encode(), "sentiment_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════
# Analyze > Website Analysis (SEO / GEO)
# ══════════════════════════════════════════════════════════════════════
elif mode == "Website Analysis":
    eyebrow("Website SEO / GEO")
    st.caption("Checklist audit of a single web page: search-engine basics (SEO), signals that help AI answer "
               "engines read and cite it (GEO), Google performance data and the tone of its copy.")

    missing = [p for p in ("requests", "bs4") if importlib.util.find_spec(p) is None]
    if missing:
        friendly_error("Website Analysis needs extra packages.",
                       "Run `pip install requests beautifulsoup4` and restart the app.")
        st.stop()
    try:
        _website = _load_module_by_path("website_analysis", "09_website_analysis.py")
    except Exception as e:
        friendly_error("Couldn't load 09_website_analysis.py.", "Check the file exists next to 03_dashboard.py.", e)
        st.stop()

    @st.cache_data(show_spinner=False, ttl=3600)
    def cached_website_analysis(url: str, run_pagespeed: bool, run_sentiment: bool) -> dict:
        return _website.analyze_url(url, run_pagespeed=run_pagespeed, run_sentiment=run_sentiment,
                                    predictor=predictor if run_sentiment else None)

    c1, c2 = st.columns([3, 1])
    with c1:
        site_url = st.text_input("Page URL", placeholder="e.g. https://www.example.com/pricing",
                                 label_visibility="collapsed")
    with c2:
        run_site = st.button("Analyze page", use_container_width=True, type="primary")
    o1, o2 = st.columns(2)
    with o1:
        use_psi = st.toggle("Include Google PageSpeed (slower, ~30-60 s)", value=True)
    with o2:
        use_sent = st.toggle("Include copy sentiment", value=True)

    if run_site and site_url.strip():
        with st.spinner("Fetching page and running checks..."):
            try:
                st.session_state["website_result"] = cached_website_analysis(site_url.strip(), use_psi, use_sent)
            except _website.WebsiteAnalysisError as e:
                st.session_state.pop("website_result", None)
                friendly_error("Couldn't analyse that page.", str(e))
            except Exception as e:
                st.session_state.pop("website_result", None)
                friendly_error("Something went wrong while analysing the page.", "Try again or another URL.", e)

    res = st.session_state.get("website_result")
    if res:
        st.caption(f"Analysed **{res['final_url']}** · {res['word_count']} words of main content · {res['fetched_at']} UTC"
                   + (f" · redirected from {res['redirect_chain'][0]}" if res["redirect_chain"] else ""))

        def tier_of(score):
            if score is None:
                return None
            return "high" if score >= 80 else "medium" if score >= 50 else "low"

        k1, k2, k3 = st.columns(3)
        for col, key, title, heur in ((k1, "seo", "SEO score", True), (k2, "geo", "GEO score", True),
                                      (k3, "performance", "Performance", False)):
            sc, t = res["scores"][key], tier_of(res["scores"][key])
            with col:
                if t is None:
                    kpi_card(title, "UNAVAILABLE", "#F1F5F9", "#475569",
                             (res["performance"].get("error") or "Not run")[:90])
                else:
                    kpi_card(title, QUALITY_LABELS[t], TIER_BG[t], TIER_COLORS[t], f"Score {sc} / 100",
                             footnote="Google Lighthouse lab score" if key == "performance" else "Weighted checklist",
                             heuristic=heur)
        st.caption(res["score_note"])

        checks = res["checks"]
        to_fix = [c for c in checks if c["status"] in ("fail", "warn")]
        eyebrow("Top fixes")
        if to_fix:
            order = {"fail": 0, "warn": 1}
            reason_card([f"<b>{c['category']} · {c['name']}</b> ({c['status']}): {c['fix']}"
                         for c in sorted(to_fix, key=lambda c: order[c["status"]])[:6]])
        else:
            info_card("Every check passed.")

        perf = res["performance"]
        if perf.get("available"):
            eyebrow(f"Core Web Vitals ({perf.get('strategy', 'mobile')})")
            v1, v2, v3 = st.columns(3)
            rating_tier = {"good": "high", "needs improvement": "medium", "poor": "low"}
            for col, k, name in ((v1, "lcp_ms", "LCP · loading"), (v2, "cls", "CLS · visual stability"),
                                 (v3, "inp_ms", "INP · responsiveness")):
                v = perf["vitals"][k]
                with col:
                    if v["value"] is None:
                        kpi_card(name, "NO DATA", "#F1F5F9", "#475569", "Not enough real-user data")
                    else:
                        t = rating_tier[v["rating"]]
                        val = f"{v['value']:.3f}" if k == "cls" else f"{v['value']:.0f} ms"
                        kpi_card(name, v["rating"].upper(), TIER_BG[t], TIER_COLORS[t], val,
                                 footnote=f"{v['source']} data")

        sent = res["sentiment"]
        eyebrow("Tone of the page copy")
        if sent.get("available"):
            s = sent["scores"]
            info_card(f"Mostly <b>{sent['label']}</b> ({sent['confidence']:.0%}) across {sent['chunks']} text chunks · "
                      f"positive {s['positive']:.0%}, neutral {s['neutral']:.0%}, negative {s['negative']:.0%}.")
        else:
            info_card(sent.get("error", "Not run."))

        with st.expander(f"All {len(checks)} checks with fix suggestions", expanded=False):
            icon = {"pass": "✅ pass", "warn": "⚠️ warn", "fail": "❌ fail", "skip": "○ skip"}
            st.dataframe(
                pd.DataFrame([{"Area": c["category"], "Check": c["name"], "Result": icon.get(c["status"], c["status"]),
                               "Value": str(c["value"]), "Details": c["detail"], "How to fix": c["fix"]}
                              for c in checks]),
                use_container_width=True, hide_index=True,
                column_config={"Details": st.column_config.TextColumn(width="large"),
                               "How to fix": st.column_config.TextColumn(width="large")},
            )
        with st.expander("How the scores are calculated"):
            st.markdown(
                "Each check scores **pass = 1, warn = 0.5, fail = 0**, multiplied by its weight; checks that "
                "couldn't run are left out. Score = weighted points ÷ weights that ran × 100.  \n"
                f"**SEO weights:** {', '.join(f'{k} {v}' for k, v in res['weights']['seo'].items())}  \n"
                f"**GEO weights:** {', '.join(f'{k} {v}' for k, v in res['weights']['geo'].items())}  \n"
                "These are checklists, not predictions of search ranking or AI citations."
            )
        st.download_button("Download full results (JSON)",
                           json.dumps(res, indent=2, default=str).encode(),
                           "website_analysis.json", "application/json")

# ══════════════════════════════════════════════════════════════════════
# Create & Edit > Generative Editing
# ══════════════════════════════════════════════════════════════════════
elif mode == "Generative Editing":
    eyebrow("Generative editing")
    st.caption("Restyle an image with a text prompt using Stable Diffusion (img2img).")
    note(
        "This is <b>generative AI</b>, not a pixel tweak — the model reinterprets the whole "
        "image from your prompt, so results vary each time. First run downloads the model "
        "(~4-5GB). On CPU this can take several minutes per image; seconds on a CUDA GPU."
    )

    if importlib.util.find_spec("diffusers") is None:
        st.warning(
            "Generative editing needs the `diffusers` package (and a working PyTorch install). "
            "Run: `pip install diffusers accelerate` and restart the app to enable this page."
        )
    else:
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="gen_upload")
        reset_state_on_new_upload(uploaded_img, "gen_edit_key", "gen_edit_output")

        if uploaded_img is not None:
            image = open_uploaded_image(uploaded_img)
            if image is not None:
                st.image(image, caption="Original", use_container_width=True)

                prompt = st.text_input("Describe the new style/scene", placeholder="e.g. on a warm sunset beach, golden hour lighting")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    strength = st.slider("Strength (how much changes)", 0.2, 0.9, 0.55, 0.05)
                with col_b:
                    steps = st.slider("Inference steps", 10, 50, 25, 5)
                with col_c:
                    guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5, 0.5)

                device_guess = "cuda" if _cuda_available() else "cpu"
                st.caption(f"Estimated time: {_generative_editing.estimate_time_warning(device_guess, steps)}")

                generate_clicked = st.button("Generate", type="primary", disabled=not prompt.strip())

                if generate_clicked:
                    try:
                        with st.spinner("Generating (this can take a while on CPU)..."):
                            editor = load_sd_editor()
                            generated = editor.generate(
                                image, prompt=prompt, strength=strength,
                                guidance_scale=guidance, num_inference_steps=steps,
                            )
                        st.session_state["gen_edit_output"] = generated
                    except ImportError as e:
                        friendly_error(
                            "Stable Diffusion isn't available.",
                            "Install the optional dependencies with `pip install diffusers accelerate`.",
                            e,
                        )
                    except Exception as e:
                        friendly_error(
                            "Image generation failed.",
                            "This can happen with no internet connection (first-time model download), "
                            "low memory, or an unsupported prompt. Try a shorter prompt or fewer steps.",
                            e,
                        )

                if "gen_edit_output" in st.session_state:
                    generated = st.session_state["gen_edit_output"]
                    clip_scorer = load_clip_scorer()
                    before = _image_analysis.score_image(image, clip_scorer=clip_scorer)
                    after = _image_analysis.score_image(generated, clip_scorer=clip_scorer)

                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.image(image, caption=f"Before (heuristic {before['score']:.0f}/100)", use_container_width=True)
                    with col_after:
                        st.image(generated, caption=f"After (heuristic {after['score']:.0f}/100)", use_container_width=True)
                    st.caption("Engagement scores above are the same heuristic used elsewhere in this app — not a validated prediction.")

                    image_download_button(generated, "Download generated image", "generative_edit.png")

# ══════════════════════════════════════════════════════════════════════
# Create & Edit > Precise Editing
# ══════════════════════════════════════════════════════════════════════
elif mode == "Precise Editing":
    eyebrow("Precise editing")
    st.caption(
        "Exact background color swaps and text replacement - deliberately not generative, "
        "so colors and text come out exact rather than approximate."
    )

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="precise_upload")
    reset_state_on_new_upload(uploaded_img, "precise_edit_key", "detected_text_regions", "precise_edit_output")

    if uploaded_img is not None:
        image = open_uploaded_image(uploaded_img)

        if image is not None:
            clip_scorer = load_clip_scorer()
            base_result = _image_analysis.score_image(image, clip_scorer=clip_scorer)
            st.image(image, caption=f"Original (heuristic engagement {base_result['score']:.0f}/100)", use_container_width=True)

            eyebrow("Background color")
            st.caption("Cuts out the subject and places it on an exact solid color - not an approximation.")
            bg_color = st.color_picker("Pick a background color", "#FFFFFF")
            swap_clicked = st.button("Swap background")

            if swap_clicked:
                try:
                    with st.spinner("Removing background (first run downloads a small model)..."):
                        swapped = _image_editing.swap_background_to_color(image, rgb_from_hex(bg_color))
                    swapped_result = _image_analysis.score_image(swapped, clip_scorer=clip_scorer)

                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.image(image, caption=f"Before ({base_result['score']:.0f}/100)", use_container_width=True)
                    with col_after:
                        st.image(swapped, caption=f"Background swapped ({swapped_result['score']:.0f}/100)", use_container_width=True)

                    image_download_button(swapped, "Download image", "background_swapped.png")
                except ImportError as e:
                    friendly_error("Background removal isn't available.", "Install it with: `pip install rembg`.", e)
                except Exception as e:
                    friendly_error("Background swap failed.", "Try a different image, or check the technical details below.", e)

            eyebrow("Text")
            st.caption("Detects text in the image so you can replace it. Works best on flat, simple backgrounds.")

            if st.button("Detect text"):
                try:
                    with st.spinner("Reading text in the image (first run downloads OCR models)..."):
                        regions = _image_editing.detect_text_regions(image)
                    st.session_state["detected_text_regions"] = regions
                    if not regions:
                        st.caption("No text detected in this image.")
                except ImportError as e:
                    friendly_error("Text detection isn't available.", "Install it with: `pip install easyocr`.", e)
                except Exception as e:
                    friendly_error("Text detection failed.", "Try a different image, or check the technical details below.", e)

            regions = st.session_state.get("detected_text_regions", [])
            if regions:
                labels = [f"\"{r['text']}\" ({r['confidence']:.0%} confidence)" for r in regions]
                chosen_idx = st.selectbox("Detected text", range(len(regions)), format_func=lambda i: labels[i])
                new_text = st.text_input("Replace with", value=regions[chosen_idx]["text"])
                text_color = st.color_picker("Text color", "#000000")

                if st.button("Replace text"):
                    edited = _image_editing.replace_text(image, regions[chosen_idx]["bbox"], new_text, text_color=rgb_from_hex(text_color))
                    edited_result = _image_analysis.score_image(edited, clip_scorer=clip_scorer)

                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.image(image, caption=f"Before ({base_result['score']:.0f}/100)", use_container_width=True)
                    with col_after:
                        st.image(edited, caption=f"Text replaced ({edited_result['score']:.0f}/100)", use_container_width=True)

                    image_download_button(edited, "Download image", "text_replaced.png")

# ══════════════════════════════════════════════════════════════════════
# Create & Edit > Canvas Editor
# ══════════════════════════════════════════════════════════════════════
else:
    eyebrow("Canvas editor")
    st.caption(
        "Draw directly on your image: freehand annotate, drag a rectangle to crop or "
        "place a text box. This is a lightweight editor, not a full design tool - "
        "for exact background swaps or OCR text replacement, use the Precise Editing page instead."
    )

    if importlib.util.find_spec("streamlit_drawable_canvas") is None:
        st.warning(
            "Canvas editor needs the `streamlit-drawable-canvas` package. "
            "Run: `pip install streamlit-drawable-canvas` and restart the app to enable this page."
        )
    else:
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image

        canvas_upload = st.file_uploader("Upload an image to edit", type=["jpg", "jpeg", "png"], key="canvas_upload")

        if canvas_upload is not None:
            base_image = open_uploaded_image(canvas_upload)
            base_image = base_image.convert("RGB") if base_image is not None else None

        if canvas_upload is not None and base_image is not None:
            # scale down for display if large - canvas gets sluggish on big images
            display_w = min(700, base_image.width)
            scale = display_w / base_image.width
            display_h = int(base_image.height * scale)
            display_image = base_image.resize((display_w, display_h))

            col_tools, col_canvas = st.columns([1, 3])
            with col_tools:
                tool = st.radio("Tool", ["freedraw", "rect", "line", "circle", "transform"])
                stroke_color = st.color_picker("Stroke color", "#FF0000")
                stroke_width = st.slider("Stroke width", 1, 15, 3)
                fill_hex = st.color_picker("Fill color (shapes)", "#FFFFFF")
                fill_opacity = st.slider("Fill opacity", 0.0, 1.0, 0.0, 0.05)
                fill_rgb = rgb_from_hex(fill_hex)
                fill_color = f"rgba({fill_rgb[0]}, {fill_rgb[1]}, {fill_rgb[2]}, {fill_opacity})"

            with col_canvas:
                canvas_result = st_canvas(
                    fill_color=fill_color,
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_image=display_image,
                    height=display_h,
                    width=display_w,
                    drawing_mode=tool,
                    key="drawable_canvas",
                )

            eyebrow("Actions")
            action_col1, action_col2, action_col3 = st.columns(3)

            def _last_rect_bbox():
                """Finds the most recently drawn rectangle and converts its
                canvas coordinates back to the original image's coordinates."""
                if canvas_result.json_data is None:
                    return None
                objects = [o for o in canvas_result.json_data["objects"] if o["type"] == "rect"]
                if not objects:
                    return None
                obj = objects[-1]
                x1 = obj["left"] / scale
                y1 = obj["top"] / scale
                x2 = (obj["left"] + obj["width"] * obj.get("scaleX", 1)) / scale
                y2 = (obj["top"] + obj["height"] * obj.get("scaleY", 1)) / scale
                return (int(x1), int(y1), int(x2), int(y2))

            with action_col1:
                if st.button("Crop to rectangle"):
                    bbox = _last_rect_bbox()
                    if bbox is None:
                        st.warning("Draw a rectangle first (select the 'rect' tool).")
                    else:
                        cropped = base_image.crop(bbox)
                        st.session_state["canvas_output"] = cropped
                        st.image(cropped, caption="Cropped", use_container_width=True)

            with action_col2:
                with st.popover("Add text box"):
                    text_content = st.text_input("Text", key="canvas_text_input")
                    text_color_hex = st.color_picker("Text color", "#000000", key="canvas_text_color")
                    box_bg_hex = st.color_picker("Box background", "#FFFFFF", key="canvas_box_bg")
                    if st.button("Place text in last rectangle"):
                        bbox = _last_rect_bbox()
                        if bbox is None:
                            st.warning("Draw a rectangle first to mark where the text goes.")
                        elif not text_content.strip():
                            st.warning("Type some text first.")
                        else:
                            edited = _image_editing.replace_text(
                                base_image, bbox, text_content,
                                text_color=rgb_from_hex(text_color_hex), fill_color=rgb_from_hex(box_bg_hex),
                            )
                            st.session_state["canvas_output"] = edited
                            st.image(edited, caption="Text added", use_container_width=True)

            with action_col3:
                if st.button("Flatten drawing to image"):
                    if canvas_result.image_data is not None:
                        drawing = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
                        drawing_full = drawing.resize(base_image.size)
                        flattened = base_image.convert("RGBA")
                        flattened = Image.alpha_composite(flattened, drawing_full).convert("RGB")
                        st.session_state["canvas_output"] = flattened
                        st.image(flattened, caption="Flattened", use_container_width=True)
                    else:
                        st.warning("Draw something first.")

            if "canvas_output" in st.session_state:
                image_download_button(st.session_state["canvas_output"], "Download edited image", "canvas_edited.png")
