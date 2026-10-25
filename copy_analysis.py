"""
copy_analysis.py
----------------
Ad-copy analysis used by the web app (03_web_app.py): keyword features,
a plain-language explanation of the sentiment prediction, the heuristic
engagement score, and the grounded rewriter (it can never invent prices,
discounts, deadlines or guarantees that the advertiser didn't supply).

Moved here unchanged from the old Streamlit dashboard, so the logic lives
in one importable module instead of inside UI code.
"""

import os
import re
from functools import lru_cache

import numpy as np

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



@lru_cache(maxsize=1)
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


@lru_cache(maxsize=2048)
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



def recommendation(label: str) -> str:
    if label == "negative":
        return ("Negative tone detected. Consider reframing pain points as solutions, using positive "
                "action verbs, and adding a clear benefit statement.")
    if label == "neutral":
        return ("Neutral tone. To boost engagement, add emotional triggers, urgency words ('now', "
                "'today', 'exclusive'), or a stronger call-to-action.")
    return "Positive tone - likely to resonate well with audiences. Pair with high-quality visuals for maximum impact."
