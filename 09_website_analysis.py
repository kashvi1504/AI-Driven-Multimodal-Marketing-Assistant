"""
09_website_analysis.py
----------------------
Single-page website audit for SEO (search engine optimisation) and GEO
(generative engine optimisation - how easy a page is for AI answer
engines such as ChatGPT, Perplexity and Google AI Overviews to read,
trust and cite).

    python 09_website_analysis.py https://example.com
    python 09_website_analysis.py https://example.com --no-pagespeed --no-sentiment
    python 09_website_analysis.py https://example.com --json results/site.json

Everything here is a CHECKLIST, not a prediction. The SEO and GEO scores
are heuristic: a weighted share of checks passed (weights documented
below). They do not predict search rankings or whether an AI engine will
actually cite the page - no ranking or citation data is used anywhere.

GEO background: Aggarwal et al., "GEO: Generative Engine Optimization"
(KDD 2024, arXiv:2311.09735) tested content changes on a benchmark of
queries answered by generative engines. Adding citations to sources,
quotations from relevant sources and statistics were among the changes
that most improved a source's visibility in the generated answers. The
GEO checks below look for those signals, plus the machine-readable
structure (schema.org JSON-LD, clear author/date, crawler access) that
lets an engine read and attribute a page in the first place.

This module does not import Streamlit, so it works from the command line
and from 03_dashboard.py ("Website Analysis" page) alike.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import MODEL_DIR, WEBSITE_ANALYSIS as CFG

HEADERS = {"User-Agent": CFG["user_agent"], "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5"}

# ── Scoring weights (each group sums to 100) ─────────────────────────
# Score = sum(weight * points) / sum(weight of checks that ran) * 100,
# where points are pass = 1, warn = 0.5, fail = 0. Checks that could not
# run ("skip") are left out of both sums rather than counted as failures.
# Weights reflect how basic/impactful each item is in common SEO
# guidance; they are a documented judgement call, not fitted to data.
SEO_WEIGHTS = {
    "title": 15,           # shown as the search result headline
    "https": 15,           # baseline trust / ranking signal
    "meta_description": 10,
    "h1": 10,
    "viewport": 10,        # mobile-friendliness (mobile-first indexing)
    "alt_text": 10,
    "broken_links": 10,
    "word_count": 10,
    "heading_order": 5,
    "canonical": 5,
}
GEO_WEIGHTS = {
    "structured_data": 20,  # machine-readable facts about the page
    "citations": 15,        # GEO paper: citing sources
    "statistics": 15,       # GEO paper: adding statistics
    "author_date": 15,      # attribution + freshness
    "ai_crawlers": 15,      # engines can't cite what they may not crawl
    "questions": 15,        # Q&A structure matches how people ask engines
    "quotations": 5,        # GEO paper: quotations
}
POINTS = {"pass": 1.0, "warn": 0.5, "fail": 0.0}

# Core Web Vitals thresholds (web.dev): (good <=, poor >)
CWV = {"lcp_ms": (2500, 4000), "cls": (0.1, 0.25), "inp_ms": (200, 500)}

QUESTION_START = re.compile(
    r"^(how|what|why|when|where|who|which|can|does|do|is|are|should|will|could|would)\b", re.I)
STAT_RE = re.compile(
    r"(?:[$€£₹]\s?\d[\d,]*(?:\.\d+)?\s?(?:k|m|bn|million|billion|crore|lakh)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent|per cent|million|billion|thousand|crore|lakh|x\b))",
    re.I)
NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
QUOTE_RE = re.compile(r"[\"“][^\"”]{25,300}[\"”]")


class WebsiteAnalysisError(Exception):
    """Raised with a message that is safe to show to a user as-is."""


# ── Helpers ──────────────────────────────────────────────────────────
def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise WebsiteAnalysisError("Please enter a URL.")
    if re.search(r"\s", url):
        raise WebsiteAnalysisError(f"'{url}' doesn't look like a website address (it contains spaces).")
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    parsed = urlparse(url)
    if not parsed.netloc or "." not in parsed.netloc.split(":")[0] and parsed.hostname != "localhost":
        raise WebsiteAnalysisError(f"'{url}' doesn't look like a website address.")
    return url


def _site_key(netloc: str) -> str:
    host = (netloc or "").lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _check(cid, category, name, status, value, detail, fix=""):
    return {"id": cid, "category": category, "name": name, "status": status,
            "value": value, "detail": detail, "fix": fix if status != "pass" else ""}


def fetch_page(url: str) -> dict:
    """GET the page, following redirects. Friendly errors for the usual
    failure modes (DNS, timeout, SSL, HTTP errors, non-HTML, too large)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=CFG["timeout"], allow_redirects=True, stream=True)
    except requests.exceptions.SSLError:
        raise WebsiteAnalysisError("The site's SSL/HTTPS certificate could not be verified.")
    except requests.exceptions.ConnectionError:
        raise WebsiteAnalysisError("Could not connect - check the address, or the site may be down.")
    except requests.exceptions.Timeout:
        raise WebsiteAnalysisError(f"The site did not respond within {CFG['timeout']} seconds.")
    except requests.exceptions.TooManyRedirects:
        raise WebsiteAnalysisError("The page redirects in a loop.")
    except requests.exceptions.RequestException as e:
        raise WebsiteAnalysisError(f"Request failed: {e.__class__.__name__}.")

    chain = [r.url for r in resp.history] + [resp.url]
    if resp.status_code >= 400:
        raise WebsiteAnalysisError(
            f"The server answered HTTP {resp.status_code} for {resp.url}"
            + (" (the site may block automated tools)." if resp.status_code in (401, 403, 429) else "."))
    ctype = resp.headers.get("Content-Type", "").lower()
    if "html" not in ctype and "xml" not in ctype:
        raise WebsiteAnalysisError(
            f"That URL returns '{ctype.split(';')[0] or 'unknown content'}', not a web page.")
    raw = resp.raw.read(CFG["max_bytes"] + 1, decode_content=True)
    if len(raw) > CFG["max_bytes"]:
        raise WebsiteAnalysisError(f"Page is larger than {CFG['max_bytes'] // 1_000_000} MB - skipped.")
    enc = resp.encoding if resp.encoding and resp.encoding.lower() != "iso-8859-1" else None
    html = raw.decode(enc or resp.apparent_encoding or "utf-8", errors="replace")
    return {"requested_url": url, "final_url": resp.url, "status_code": resp.status_code,
            "redirect_chain": chain if len(chain) > 1 else [], "content_type": ctype.split(";")[0],
            "html": html, "bytes": len(raw)}


def extract_main_text(soup: BeautifulSoup) -> str:
    """Visible body copy: prefer <main>/<article>, drop nav/footer/scripts."""
    work = BeautifulSoup(str(soup), "html.parser")
    for tag in work(["script", "style", "noscript", "svg", "template", "iframe", "form",
                     "nav", "footer", "header", "aside"]):
        tag.decompose()
    root = work.find("main") or work.find("article") or work.body or work
    text = root.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def _jsonld_items(soup):
    items, errors = [], 0
    for tag in soup.find_all("script", type=lambda t: t and "ld+json" in t.lower()):
        try:
            data = json.loads(tag.string or tag.get_text() or "")
        except (json.JSONDecodeError, TypeError):
            errors += 1
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                items.append(node)
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
    return items, errors


def _types(items):
    out = set()
    for it in items:
        t = it.get("@type")
        for x in (t if isinstance(t, list) else [t]):
            if isinstance(x, str):
                out.add(x.split("/")[-1])
    return sorted(out)


# ── SEO checks ───────────────────────────────────────────────────────
def check_links(soup, base_url):
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            continue
        absu = urljoin(base_url, href).split("#")[0]
        if urlparse(absu).scheme in ("http", "https") and absu not in links:
            links.append(absu)
    sample = links[: CFG["max_link_checks"]]

    def probe(u):
        try:
            r = requests.head(u, headers=HEADERS, timeout=CFG["timeout"], allow_redirects=True)
            if r.status_code in (403, 405, 501) or r.status_code >= 500:
                r = requests.get(u, headers=HEADERS, timeout=CFG["timeout"], allow_redirects=True, stream=True)
                r.close()
            return u, r.status_code
        except requests.exceptions.RequestException as e:
            return u, e.__class__.__name__

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(probe, sample))
    # 401/403/429 usually mean "bot blocked", not "broken" - reported separately
    broken = [(u, s) for u, s in results if isinstance(s, str) or (s >= 400 and s not in (401, 403, 429))]
    blocked = [(u, s) for u, s in results if not isinstance(s, str) and s in (401, 403, 429)]
    return {"total_links": len(links), "checked": len(sample), "broken": broken, "blocked": blocked}


def seo_checks(soup, page, main_text, link_report):
    C = []
    title = (soup.title.get_text(strip=True) if soup.title else "")
    n = len(title)
    if not title:
        C.append(_check("title", "SEO", "Title tag", "fail", "missing", "No <title> tag.",
                        "Add a unique, descriptive <title> of 30-60 characters with the main keyword first."))
    else:
        ok = 30 <= n <= 60
        C.append(_check("title", "SEO", "Title tag", "pass" if ok else "warn", f"{n} chars",
                        f"\"{title[:90]}\"", "Aim for 30-60 characters so it isn't cut off in results"
                        if not ok else ""))

    md = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    desc = (md.get("content") or "").strip() if md else ""
    n = len(desc)
    if not desc:
        C.append(_check("meta_description", "SEO", "Meta description", "fail", "missing",
                        "No meta description.", "Add a 70-160 character summary that invites the click."))
    else:
        ok = 70 <= n <= 160
        C.append(_check("meta_description", "SEO", "Meta description", "pass" if ok else "warn", f"{n} chars",
                        f"\"{desc[:120]}\"", "Keep it between 70 and 160 characters." if not ok else ""))

    h1s = [h.get_text(" ", strip=True) for h in soup.find_all("h1")]
    st = "pass" if len(h1s) == 1 else "fail" if not h1s else "warn"
    C.append(_check("h1", "SEO", "Single H1", st, f"{len(h1s)} found",
                    "; ".join(f"\"{h[:60]}\"" for h in h1s[:3]) or "No <h1>.",
                    {"fail": "Add exactly one <h1> stating the page topic.",
                     "warn": "Keep one <h1>; turn the others into <h2>."}.get(st, "")))

    levels = [int(h.name[1]) for h in soup.find_all(re.compile(r"^h[1-6]$"))]
    skips = [f"h{a}→h{b}" for a, b in zip(levels, levels[1:]) if b > a + 1]
    if not levels:
        C.append(_check("heading_order", "SEO", "Heading order", "fail", "no headings",
                        "The page has no h1-h6 headings.", "Structure content with h1 → h2 → h3 headings."))
    else:
        st = "pass" if not skips and levels[0] == 1 else "warn" if len(skips) <= 2 else "fail"
        C.append(_check("heading_order", "SEO", "Heading order", st, f"{len(skips)} skipped level(s)",
                        (f"Skips: {', '.join(skips[:5])}. " if skips else "")
                        + f"First heading is h{levels[0]}.",
                        "Don't jump heading levels (e.g. h2 → h4); start with the h1."))

    imgs = soup.find_all("img")
    if not imgs:
        C.append(_check("alt_text", "SEO", "Image alt text", "pass", "no images", "No <img> tags on the page."))
    else:
        with_alt = sum(1 for i in imgs if i.has_attr("alt"))  # alt="" is valid for decorative images
        pct = 100 * with_alt / len(imgs)
        st = "pass" if pct >= 90 else "warn" if pct >= 50 else "fail"
        C.append(_check("alt_text", "SEO", "Image alt text", st, f"{pct:.0f}%",
                        f"{with_alt} of {len(imgs)} images have an alt attribute.",
                        "Describe each meaningful image in its alt attribute (alt=\"\" for purely decorative ones)."))

    can = soup.find("link", rel=lambda r: r and "canonical" in [x.lower() for x in (r if isinstance(r, list) else [r])])
    href = can.get("href", "").strip() if can else ""
    C.append(_check("canonical", "SEO", "Canonical tag", "pass" if href else "warn",
                    urljoin(page["final_url"], href) if href else "missing",
                    "Tells search engines which URL is the original." if href else "No rel=canonical link.",
                    "Add <link rel=\"canonical\" href=\"...\"> to avoid duplicate-URL issues."))

    vp = soup.find("meta", attrs={"name": re.compile("^viewport$", re.I)})
    C.append(_check("viewport", "SEO", "Mobile viewport", "pass" if vp else "fail",
                    (vp.get("content") or "")[:60] if vp else "missing",
                    "Viewport meta tag present." if vp else "No viewport meta tag - page will render zoomed-out on phones.",
                    "Add <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">."))

    https = urlparse(page["final_url"]).scheme == "https"
    C.append(_check("https", "SEO", "HTTPS", "pass" if https else "fail", "yes" if https else "no",
                    f"Final URL: {page['final_url']}", "Serve the site over HTTPS and redirect http:// to it."))

    lr = link_report
    nb = len(lr["broken"])
    if lr["checked"] == 0:
        C.append(_check("broken_links", "SEO", "Broken links", "pass", "no links", "No links to check."))
    else:
        st = "pass" if nb == 0 else "warn" if nb <= 2 else "fail"
        detail = f"Checked {lr['checked']} of {lr['total_links']} links; {nb} broken"
        if lr["blocked"]:
            detail += f", {len(lr['blocked'])} refused automated checks (401/403/429, not counted)"
        if nb:
            detail += ": " + "; ".join(f"{u} ({s})" for u, s in lr["broken"][:5])
        C.append(_check("broken_links", "SEO", "Broken links", st, f"{nb} broken", detail + ".",
                        "Fix or remove links that return errors."))

    wc = len(main_text.split())
    st = "pass" if wc >= 300 else "warn" if wc >= 100 else "fail"
    C.append(_check("word_count", "SEO", "Word count", st, f"{wc} words",
                    "Visible main-content words (nav/footer excluded).",
                    "Thin pages rank poorly - add useful, specific content (300+ words is a common guideline)."))
    return C


# ── GEO checks ───────────────────────────────────────────────────────
def check_robots(final_url):
    parsed = urlparse(final_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        r = requests.get(robots_url, headers=HEADERS, timeout=CFG["timeout"])
    except requests.exceptions.RequestException as e:
        return {"url": robots_url, "error": e.__class__.__name__}
    if r.status_code in (404, 410):
        return {"url": robots_url, "missing": True, "blocked": [], "allowed": CFG["ai_crawlers"]}
    if r.status_code >= 400:
        return {"url": robots_url, "error": f"HTTP {r.status_code}"}
    rp = robotparser.RobotFileParser()
    rp.parse(r.text.splitlines())
    blocked = [a for a in CFG["ai_crawlers"] if not rp.can_fetch(a, final_url)]
    return {"url": robots_url, "blocked": blocked,
            "allowed": [a for a in CFG["ai_crawlers"] if a not in blocked]}


def geo_checks(soup, page, main_text, robots):
    C = []
    items, bad = _jsonld_items(soup)
    types = _types(items)
    microdata = bool(soup.find(attrs={"itemtype": True}))
    if types:
        C.append(_check("structured_data", "GEO", "Structured data (JSON-LD)", "pass", ", ".join(types[:8]),
                        f"{len(types)} schema.org type(s) found" + (f"; {bad} block(s) failed to parse." if bad else ".")))
    else:
        C.append(_check("structured_data", "GEO", "Structured data (JSON-LD)", "warn" if microdata else "fail",
                        "microdata only" if microdata else "none",
                        ("Only microdata found. " if microdata else "No schema.org JSON-LD. ")
                        + (f"{bad} JSON-LD block(s) are invalid JSON." if bad else ""),
                        "Add schema.org JSON-LD (e.g. Organization, Article, Product, FAQPage) describing the page."))

    headings = [h.get_text(" ", strip=True) for h in soup.find_all(re.compile(r"^h[1-6]$"))]
    qs = [h for h in headings if h.endswith("?") or QUESTION_START.match(h)]
    has_faq = any(t in ("FAQPage", "QAPage") for t in types)
    st = "pass" if has_faq or len(qs) >= 2 else "warn"
    C.append(_check("questions", "GEO", "FAQ / question headings", st,
                    f"{len(qs)} question heading(s)" + (" + FAQPage schema" if has_faq else ""),
                    "; ".join(f"\"{q[:60]}\"" for q in qs[:3]) or "No question-style headings.",
                    "Add an FAQ section with real customer questions as headings and direct answers."))

    wc = max(len(main_text.split()), 1)
    stats = STAT_RE.findall(main_text)
    nums = NUMBER_RE.findall(main_text)
    per100 = 100 * len(nums) / wc
    st = "pass" if len(stats) >= 3 or (len(nums) >= 8 and per100 >= 1) else "warn" if nums else "fail"
    C.append(_check("statistics", "GEO", "Statistics & numbers", st,
                    f"{len(stats)} stats, {len(nums)} numbers",
                    f"{per100:.1f} numbers per 100 words. " + (f"e.g. {', '.join(s.strip() for s in stats[:4])}" if stats else ""),
                    "Back claims with specific figures (percentages, amounts, dates) and say where they come from."))

    here = _site_key(urlparse(page["final_url"]).netloc)
    ext = set()
    for a in soup.find_all("a", href=True):
        u = urljoin(page["final_url"], a["href"])
        p = urlparse(u)
        if p.scheme in ("http", "https") and _site_key(p.netloc) not in ("", here) \
                and not _site_key(p.netloc).endswith("." + here):
            ext.add(_site_key(p.netloc))
    st = "pass" if len(ext) >= 3 else "warn" if ext else "fail"
    C.append(_check("citations", "GEO", "Outbound citations", st, f"{len(ext)} external domain(s)",
                    ", ".join(sorted(ext)[:6]) or "No links to other sites.",
                    "Cite and link authoritative sources for your claims (studies, official data, experts)."))

    quotes = len(soup.find_all(["blockquote", "q"])) + len(QUOTE_RE.findall(main_text))
    C.append(_check("quotations", "GEO", "Quotations", "pass" if quotes else "warn", f"{quotes} found",
                    "Quoted statements (blockquote/q tags or quoted sentences).",
                    "Where relevant, include attributed quotes from experts or customers."))

    def meta(*names):
        for nm in names:
            m = soup.find("meta", attrs={"name": nm}) or soup.find("meta", attrs={"property": nm})
            if m and m.get("content"):
                return m["content"].strip()
        return ""

    author = meta("author", "article:author", "twitter:creator") \
        or next((str(it["author"].get("name") if isinstance(it["author"], dict) else it["author"])
                 for it in items if it.get("author")), "") \
        or (soup.find(attrs={"rel": "author"}) or soup.find(attrs={"itemprop": "author"}) or BeautifulSoup("", "html.parser")).get_text(strip=True)
    date = meta("article:published_time", "article:modified_time", "date", "dc.date", "last-modified") \
        or next((str(it.get("dateModified") or it.get("datePublished"))
                 for it in items if it.get("dateModified") or it.get("datePublished")), "") \
        or ((soup.find("time") or {}).get("datetime", "") if soup.find("time") else "")
    found = [x for x in ("author", "date") if (author if x == "author" else date)]
    st = "pass" if len(found) == 2 else "warn" if found else "fail"
    C.append(_check("author_date", "GEO", "Author & date", st,
                    " + ".join(found) or "neither",
                    f"Author: {str(author)[:50] or '—'}; date: {str(date)[:30] or '—'}.",
                    "Show who wrote/owns the content and when it was published or updated (visible and in metadata)."))

    if robots.get("error"):
        C.append(_check("ai_crawlers", "GEO", "AI crawler access (robots.txt)", "skip", "unknown",
                        f"Could not read {robots['url']} ({robots['error']})."))
    else:
        b = robots["blocked"]
        st = "pass" if not b else "fail" if len(b) == len(CFG["ai_crawlers"]) else "warn"
        detail = ("No robots.txt, so all crawlers are allowed." if robots.get("missing")
                  else f"Blocked: {', '.join(b) or 'none'}. Allowed: {', '.join(robots['allowed']) or 'none'}.")
        C.append(_check("ai_crawlers", "GEO", "AI crawler access (robots.txt)", st, f"{len(b)} blocked",
                        detail, "If you want to appear in AI answers, don't disallow these user-agents in robots.txt."))
    return C


# ── Performance (Google PageSpeed Insights) ──────────────────────────
def pagespeed(url: str) -> dict:
    params = {"url": url, "strategy": CFG["pagespeed_strategy"], "category": "performance"}
    if CFG["pagespeed_api_key"]:
        params["key"] = CFG["pagespeed_api_key"]
    try:
        r = requests.get("https://www.googleapis.com/pagespeedonline/v5/runPagespeed",
                         params=params, timeout=CFG["pagespeed_timeout"])
        data = r.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        return {"available": False, "error": f"PageSpeed Insights unreachable ({e.__class__.__name__})."}
    if r.status_code != 200 or "lighthouseResult" not in data:
        msg = (data.get("error") or {}).get("message", f"HTTP {r.status_code}") if isinstance(data, dict) else ""
        hint = " Set PAGESPEED_API_KEY to avoid the shared rate limit." if r.status_code == 429 else ""
        return {"available": False, "error": f"PageSpeed Insights: {msg[:160]}.{hint}"}

    lh = data["lighthouseResult"]
    audits = lh.get("audits", {})
    field = (data.get("loadingExperience") or {}).get("metrics", {})

    def fval(key):
        m = field.get(key)
        return m.get("percentile") if m else None

    score = (lh.get("categories", {}).get("performance", {}) or {}).get("score")
    out = {
        "available": True, "strategy": CFG["pagespeed_strategy"],
        "score": round(score * 100) if score is not None else None,
        "lab": {"lcp_ms": (audits.get("largest-contentful-paint") or {}).get("numericValue"),
                "cls": (audits.get("cumulative-layout-shift") or {}).get("numericValue")},
        # field = real Chrome users (CrUX); only exists for sites with enough traffic
        "field": {"lcp_ms": fval("LARGEST_CONTENTFUL_PAINT_MS"),
                  "cls": (fval("CUMULATIVE_LAYOUT_SHIFT_SCORE") / 100) if fval("CUMULATIVE_LAYOUT_SHIFT_SCORE") is not None else None,
                  "inp_ms": fval("INTERACTION_TO_NEXT_PAINT")},
    }

    vitals = {}
    for k in ("lcp_ms", "cls", "inp_ms"):
        v = out["field"].get(k)
        src = "field"
        if v is None:
            v, src = out["lab"].get(k), "lab"
        if v is None:
            vitals[k] = {"value": None, "rating": "n/a", "source": "none"}
            continue
        good, poor = CWV[k]
        vitals[k] = {"value": v, "source": src,
                     "rating": "good" if v <= good else "poor" if v > poor else "needs improvement"}
    out["vitals"] = vitals
    return out


# ── Sentiment (existing project model) ───────────────────────────────
_PREDICTOR = None


def load_predictor():
    """Load SentimentPredictor from 02_model_training.py by file path (the
    module name starts with a digit, so it can't be imported normally)."""
    global _PREDICTOR
    if _PREDICTOR is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_model_training.py")
        spec = importlib.util.spec_from_file_location("model_training", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PREDICTOR = mod.SentimentPredictor(MODEL_DIR)
    return _PREDICTOR


def page_sentiment(main_text: str, predictor) -> dict:
    """Average the model's class scores over up to 20 chunks of ~60 words
    (the model reads at most 128 tokens at a time)."""
    words = main_text.split()
    chunks = [" ".join(words[i:i + 60]) for i in range(0, min(len(words), 1200), 60)]
    chunks = [c for c in chunks if len(c.split()) >= 5]
    if not chunks:
        return {"available": False, "error": "Not enough text to classify."}
    preds = predictor.predict(chunks)
    avg = {k: round(sum(p["scores"].get(k, 0) for p in preds) / len(preds), 4)
           for k in ("negative", "neutral", "positive")}
    label = max(avg, key=avg.get)
    return {"available": True, "label": label, "confidence": avg[label], "scores": avg,
            "chunks": len(chunks), "model": "public fallback" if getattr(predictor, "using_fallback", False) else "fine-tuned"}


# ── Scoring + entry point ────────────────────────────────────────────
def heuristic_score(checks, weights):
    num = den = 0.0
    for c in checks:
        w = weights.get(c["id"])
        if w is None or c["status"] not in POINTS:
            continue
        num += w * POINTS[c["status"]]
        den += w
    return round(100 * num / den) if den else None


def analyze_url(url: str, run_pagespeed: bool = True, run_sentiment: bool = True, predictor=None) -> dict:
    """Audit one page. Raises WebsiteAnalysisError (friendly message) if
    the page itself can't be fetched; every optional part (PageSpeed,
    sentiment, robots.txt) degrades to 'unavailable' instead of raising."""
    typed_scheme = bool(re.match(r"^\s*https?://", url or "", re.I))
    url = normalize_url(url)
    try:
        page = fetch_page(url)
    except WebsiteAnalysisError:
        if typed_scheme:
            raise
        # user typed "example.com": we tried https:// first, now try plain http://
        url = "http://" + url[len("https://"):]
        page = fetch_page(url)
    soup = BeautifulSoup(page["html"], "html.parser")
    main_text = extract_main_text(soup)

    links = check_links(soup, page["final_url"])
    robots = check_robots(page["final_url"])
    seo = seo_checks(soup, page, main_text, links)
    geo = geo_checks(soup, page, main_text, robots)

    perf = pagespeed(page["final_url"]) if run_pagespeed else {"available": False, "error": "Skipped (PageSpeed turned off)."}

    if not run_sentiment:
        sent = {"available": False, "error": "Skipped (sentiment turned off)."}
    else:
        try:
            sent = page_sentiment(main_text, predictor or load_predictor())
        except Exception as e:  # model/torch missing or failed - never break the audit
            sent = {"available": False, "error": f"Sentiment model unavailable ({e.__class__.__name__})."}

    return {
        "url": page["requested_url"], "final_url": page["final_url"],
        "redirect_chain": page["redirect_chain"], "status_code": page["status_code"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "page_bytes": page["bytes"], "word_count": len(main_text.split()),
        "scores": {"seo": heuristic_score(seo, SEO_WEIGHTS), "geo": heuristic_score(geo, GEO_WEIGHTS),
                   "performance": perf.get("score") if perf.get("available") else None},
        "score_note": ("SEO and GEO scores are heuristic checklist scores (weighted share of checks "
                       "passed), not predictions of search ranking or AI citation."),
        "weights": {"seo": SEO_WEIGHTS, "geo": GEO_WEIGHTS},
        "checks": seo + geo, "performance": perf, "sentiment": sent,
        "links": {"total": links["total_links"], "checked": links["checked"],
                  "broken": links["broken"], "blocked": links["blocked"]},
        "robots": robots,
    }


# ── CLI report ───────────────────────────────────────────────────────
ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}


def format_report(r: dict) -> str:
    s = r["scores"]
    L = [f"Website audit: {r['final_url']}"]
    if r["redirect_chain"]:
        L.append("Redirects: " + " -> ".join(r["redirect_chain"]))
    L += [f"Fetched {r['fetched_at']} · {r['word_count']} words of main content", "",
          f"SEO score  {s['seo'] if s['seo'] is not None else '-':>4} / 100   (heuristic)",
          f"GEO score  {s['geo'] if s['geo'] is not None else '-':>4} / 100   (heuristic)",
          f"Performance {s['performance'] if s['performance'] is not None else 'unavailable':>3}"
          + (" / 100  (Google Lighthouse)" if s["performance"] is not None else ""), ""]
    for cat in ("SEO", "GEO"):
        L.append(f"── {cat} checks " + "─" * 40)
        for c in (c for c in r["checks"] if c["category"] == cat):
            L.append(f"[{ICON[c['status']]}] {c['name']}: {c['value']}")
            L.append(f"        {c['detail']}")
            if c["fix"]:
                L.append(f"        Fix: {c['fix']}")
        L.append("")
    p = r["performance"]
    L.append("── Performance " + "─" * 40)
    if p.get("available"):
        for k, name in (("lcp_ms", "LCP"), ("cls", "CLS"), ("inp_ms", "INP")):
            v = p["vitals"][k]
            val = "-" if v["value"] is None else (f"{v['value']:.0f} ms" if k != "cls" else f"{v['value']:.3f}")
            L.append(f"{name:4} {val:>10}  {v['rating']}  ({v['source']} data)")
    else:
        L.append(f"Unavailable: {p.get('error')}")
    L += ["", "── Page sentiment (project model) " + "─" * 22]
    se = r["sentiment"]
    L.append(f"{se['label']} ({se['confidence']:.0%}, {se['chunks']} chunks, {se['model']} model)"
             if se.get("available") else f"Unavailable: {se.get('error')}")
    L += ["", r["score_note"]]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Single-page SEO/GEO checklist audit.")
    ap.add_argument("url")
    ap.add_argument("--no-pagespeed", action="store_true", help="skip the Google PageSpeed Insights call")
    ap.add_argument("--no-sentiment", action="store_true", help="skip the sentiment model")
    ap.add_argument("--json", metavar="PATH", help="also save the full result as JSON")
    a = ap.parse_args()
    try:
        result = analyze_url(a.url, run_pagespeed=not a.no_pagespeed, run_sentiment=not a.no_sentiment)
    except WebsiteAnalysisError as e:
        print(f"Could not analyse {a.url}: {e}")
        sys.exit(1)
    print(format_report(result))
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved -> {a.json}")


if __name__ == "__main__":
    main()
