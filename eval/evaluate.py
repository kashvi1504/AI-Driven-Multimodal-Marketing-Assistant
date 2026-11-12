"""
eval/evaluate.py
----------------
Measures the web app's analyses against the hand-labelled test sets and
writes eval/results.md (+ eval/results.json).

    python eval/evaluate.py                 # held-out "test" split (the honest number)
    python eval/evaluate.py --split dev     # the split used while tuning
    python eval/evaluate.py --skip-images   # copy only

Copy:   eval/copy_test_set.csv  (build: python eval/build_copy_test_set.py)
Images: data/eval/image_manifest.csv (build: python eval/build_image_test_set.py)
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config  # noqa: E402
import copy_analysis as ca  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DATA = os.path.join(config.DATA_DIR, "eval")
LABELS = ["negative", "neutral", "positive"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── Copy ─────────────────────────────────────────────────────────────────
def evaluate_copy(split):
    df = pd.read_csv(os.path.join(HERE, "copy_test_set.csv"))
    df = df[df.split == split] if split != "all" else df
    mt = load("model_training", "02_model_training.py")
    models = {"Fine-tuned DistilBERT (ours)": config.MODEL_DIR,
              "Twitter RoBERTa (app default)": "cardiffnlp/twitter-roberta-base-sentiment-latest"}
    out = {"rows": len(df), "split": split, "models": {}}
    for name, path in models.items():
        predictor = mt.SentimentPredictor(path)
        t = time.time()
        results = predictor.predict(df.text.tolist())
        pred = [r["label"] for r in results]
        cm = confusion_matrix(df.label, pred, labels=LABELS)
        out["models"][name] = {
            "accuracy": accuracy_score(df.label, pred), "macro_f1": f1_score(df.label, pred, average="macro"),
            "confusion": cm.tolist(), "seconds": time.time() - t,
            "wrong": [{"text": x, "true": y, "pred": p} for x, y, p in zip(df.text, df.label, pred) if y != p],
        }
        if "RoBERTa" in name:
            df = df.assign(pred=pred, eng=[ca.predict_engagement(ca.analyze_text_features(x), r)["engagement_score"]
                                           for x, r in zip(df.text, results)])
    for fn, col in ((ca.find_cta, "has_cta"), (ca.find_urgency, "has_urgency")):
        pred = [int(bool(fn(x))) for x in df.text]
        out[col] = {"precision": precision_score(df[col], pred), "recall": recall_score(df[col], pred),
                    "f1": f1_score(df[col], pred)}
    out["engagement_by_label"] = df.groupby("label").eng.mean().round(1).to_dict()
    return out


# ── Images ───────────────────────────────────────────────────────────────
EXPECT = {  # category -> (rule, description)
    "clean": (lambda r: r["score"] >= 70, "Good (>= 70)"),
    "clean_lowres": (lambda r: r["score"] >= 45, "at least Fair (>= 45)"),
    "blur": (lambda r: r["score"] < 45, "Needs work (< 45)"),
    "tiny": (lambda r: r["score"] < 45, "Needs work (< 45)"),
    "dark": (lambda r: r["score"] < 45, "Needs work (< 45)"),
    "bright": (lambda r: r["score"] < 45, "Needs work (< 45)"),
    "noise": (lambda r: r["score"] < 70, "not Good (< 70)"),
    "banner": (lambda r: r["score"] >= 45, "at least Fair (>= 45)"),
    "wm_tiled": (lambda r: bool(r.get("watermark", {}).get("detected")) and r["score"] < 70, "watermark flagged, < 70"),
    "wm_center": (lambda r: bool(r.get("watermark", {}).get("detected")) and r["score"] < 70, "watermark flagged, < 70"),
    "wm_corner": (lambda r: bool(r.get("watermark", {}).get("detected")) and r["score"] < 70, "watermark flagged, < 70"),
}


def evaluate_images():
    manifest = pd.read_csv(os.path.join(EVAL_DATA, "image_manifest.csv"))
    ia = load("image_analysis", "04_image_analysis.py")
    clip = ia.ClipScorer()
    try:
        watermark = ia.WatermarkDetector()
    except Exception:  # noqa: BLE001 - easyocr not installed
        watermark = None
    ai = ia.AIImageDetector()
    has_wm, has_ai = watermark is not None, True
    rows = []
    t0 = time.time()
    for _, m in manifest.iterrows():
        path = os.path.join(EVAL_DATA, m.file)
        with open(path, "rb") as fh:
            raw = fh.read()
        r = ia.score_image(Image.open(path), clip_scorer=clip, watermark_detector=watermark, ai_detector=ai, raw_bytes=raw)
        rec = {"file": os.path.basename(m.file), "category": m.category, "score": round(r["score"], 1), "label": r["label"],
               "watermark": r.get("watermark") or {}, "ai": r.get("ai_generated") or {}}
        rule = EXPECT.get(m.category)
        rec["pass"] = bool(rule[0](rec)) if rule else None
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = {"images": len(df), "seconds_per_image": (time.time() - t0) / max(1, len(df)),
           "watermark_feature": has_wm, "ai_feature": has_ai, "by_category": {}}
    for cat, g in df.groupby("category"):
        entry = {"n": len(g), "mean_score": round(g.score.mean(), 1), "min": g.score.min(), "max": g.score.max()}
        if cat in EXPECT:
            entry.update(expected=EXPECT[cat][1], passed=int(g["pass"].sum()))
        out["by_category"][cat] = entry
    scored = df[df["pass"].notna()]
    out["quality_pass_rate"] = float(scored["pass"].mean())
    out["realworld"] = df[df.category == "realworld"][["file", "score", "label"]].to_dict("records")
    if has_ai:
        is_ai = df.category == "ai"
        # images whose origin we can't vouch for are left out of the "real" side
        unknown = df.file.str.contains("Storyboards|WhatsApp")
        real = df.category.isin(["clean", "clean_lowres", "realworld"]) & ~unknown
        flagged = df.ai.apply(lambda a: bool(a.get("likely")))
        out["ai_detection"] = {"ai_images": int(is_ai.sum()), "ai_flagged": int((flagged & is_ai).sum()),
                               "real_images": int(real.sum()), "real_flagged": int((flagged & real).sum())}
    if has_wm:
        wm_cats = df.category.str.startswith("wm_")
        flagged = df.watermark.apply(lambda w: bool(w.get("detected")))
        clean_like = df.category.isin(["clean", "clean_lowres", "blur", "tiny", "dark", "bright", "noise", "banner", "ai"])
        out["watermark_detection"] = {"watermarked": int(wm_cats.sum()), "flagged": int((flagged & wm_cats).sum()),
                                      "unwatermarked": int(clean_like.sum()), "false_flags": int((flagged & clean_like).sum())}
    out["failures"] = df[df["pass"] == False][["file", "category", "score"]].to_dict("records")  # noqa: E712
    return out


# ── Report ───────────────────────────────────────────────────────────────
def pct(x):
    return f"{100 * x:.0f}%"


def write_report(res):
    L = ["# Evaluation results", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python eval/evaluate.py`.", ""]
    c = res.get("copy")
    if c:
        L += [f"## Ad copy sentiment ({c['rows']} hand-labelled ads, `{c['split']}` split)", "",
              "| Model | Accuracy | Macro-F1 |", "|---|---|---|"]
        for name, m in c["models"].items():
            L.append(f"| {name} | {pct(m['accuracy'])} | {m['macro_f1']:.2f} |")
        for name, m in c["models"].items():
            L += ["", f"**{name}** confusion (rows = true label, columns = predicted):", "",
                  "| | negative | neutral | positive |", "|---|---|---|---|"]
            L += [f"| {lab} | " + " | ".join(str(v) for v in row) + " |" for lab, row in zip(LABELS, m["confusion"])]
        L += ["", "| Detector | Precision | Recall | F1 |", "|---|---|---|---|"]
        for col, name in (("has_cta", "Call to action"), ("has_urgency", "Urgency")):
            d = c[col]
            L.append(f"| {name} | {pct(d['precision'])} | {pct(d['recall'])} | {d['f1']:.2f} |")
        L += ["", "Average heuristic engagement score by true tone: " +
              ", ".join(f"{k} {v}" for k, v in c["engagement_by_label"].items()), ""]
        wrong = c["models"]["Twitter RoBERTa (app default)"]["wrong"]
        if wrong:
            L += ["Ads the app default still gets wrong:", ""]
            L += [f"- *{w['true']} → {w['pred']}*: {w['text']}" for w in wrong]
            L.append("")
    im = res.get("images")
    if im:
        L += [f"## Images ({im['images']} images)", "",
              f"Expected-verdict pass rate: **{pct(im['quality_pass_rate'])}** · "
              f"{im['seconds_per_image']:.2f} s per image", "",
              "| Category | Images | Expected | Passed | Mean score | Range |", "|---|---|---|---|---|---|"]
        for cat, e in im["by_category"].items():
            L.append(f"| {cat} | {e['n']} | {e.get('expected', 'no fixed verdict')} | {e.get('passed', '-')} | "
                     f"{e['mean_score']} | {e['min']:.0f}-{e['max']:.0f} |")
        if "watermark_detection" in im:
            w = im["watermark_detection"]
            L += ["", f"**Watermark detection:** flagged {w['flagged']}/{w['watermarked']} watermarked images; "
                      f"false alarms on {w['false_flags']}/{w['unwatermarked']} images without a watermark."]
        if "ai_detection" in im:
            a = im["ai_detection"]
            L += ["", f"**AI-generated indicator:** flagged {a['ai_flagged']}/{a['ai_images']} AI images; "
                      f"false alarms on {a['real_flagged']}/{a['real_images']} real images."]
        if im["realworld"]:
            L += ["", "Real ads scored as-is: " + "; ".join(f"{r['file']} {r['score']:.0f} ({r['label']})" for r in im["realworld"])]
        if im["failures"]:
            L += ["", "Failures: " + "; ".join(f"{f['file']} ({f['category']}, {f['score']:.0f})" for f in im["failures"])]
    with open(os.path.join(HERE, "results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-copy", action="store_true")
    args = ap.parse_args()
    res = {}
    if not args.skip_copy:
        res["copy"] = evaluate_copy(args.split)
    if not args.skip_images:
        res["images"] = evaluate_images()
    write_report(res)


if __name__ == "__main__":
    main()
