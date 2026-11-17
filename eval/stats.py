"""
eval/stats.py
-------------
Uncertainty and significance for the numbers in eval/results.md, so that
small test sets are reported honestly. Writes eval/stats.md (+ stats.json).

    python eval/stats.py

Ad copy (90 held-out ads, and all 180 as a supplementary check):
  - accuracy, macro precision / recall / F1 and per-class P / R / F1
  - 95% bootstrap confidence intervals (10,000 resamples of the ads, seed 0)
  - exact McNemar tests between models (on the ads where they disagree)
  - VADER, a widely used off-the-shelf sentiment tool, as an external baseline
  - call-to-action and urgency detectors with bootstrap CIs
Images: 95% Wilson intervals for every pass rate in results.json and
results_heldout.json.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as ev  # noqa: E402  (reuses its model loading)

LABELS = ev.LABELS
B = 10_000
rng = np.random.default_rng(0)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot(metric, y, p, idx):
    vals = np.array([metric(y[i], p[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def copy_metrics(y, p, idx):
    acc = lambda a, b: float(np.mean(a == b))
    mf1 = lambda a, b: f1_score(a, b, labels=LABELS, average="macro", zero_division=0)
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    mP, mR, mF, _ = precision_recall_fscore_support(y, p, labels=LABELS, average="macro", zero_division=0)
    out = {"accuracy": acc(y, p), "accuracy_ci": boot(acc, y, p, idx),
           "accuracy_wilson": wilson(int((y == p).sum()), len(y)),
           "macro_precision": float(mP), "macro_recall": float(mR), "macro_f1": float(mF),
           "macro_f1_ci": boot(mf1, y, p, idx), "per_class": {}}
    for j, lab in enumerate(LABELS):
        f = lambda a, b, lab=lab: f1_score(a, b, labels=[lab], average="macro", zero_division=0)
        out["per_class"][lab] = {"precision": float(P[j]), "recall": float(R[j]), "f1": float(F[j]), "f1_ci": boot(f, y, p, idx)}
    return out


def mcnemar(y, a, b):
    a_ok, b_ok = a == y, b == y
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"only_first_correct": n01, "only_second_correct": n10, "p_value": float(pv)}


def vader_labels(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        c = sia.polarity_scores(t)["compound"]  # standard thresholds from the VADER authors
        out.append("positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral")
    return out


def predictions(df):
    mt = ev.load("model_training", "02_model_training.py")
    preds = {}
    for name, path in (("Fine-tuned DistilBERT (ours)", ev.config.MODEL_DIR),
                       ("Twitter RoBERTa (app default)", "cardiffnlp/twitter-roberta-base-sentiment-latest")):
        preds[name] = np.array([r["label"] for r in mt.SentimentPredictor(path).predict(df.text.tolist())])
    preds["VADER (off-the-shelf baseline)"] = np.array(vader_labels(df.text.tolist()))
    return preds


def copy_block(df, preds):
    y = df.label.to_numpy()
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    res = {"rows": len(y), "models": {n: copy_metrics(y, p, idx) for n, p in preds.items()}}
    names = list(preds)
    rob, dist, vad = names[1], names[0], names[2]
    res["mcnemar"] = {f"{rob} vs {dist}": mcnemar(y, preds[rob], preds[dist]),
                      f"{rob} vs {vad}": mcnemar(y, preds[rob], preds[vad])}
    return res, idx


def detector_block(df, idx):
    out = {}
    for fn, col in ((ev.ca.find_cta, "has_cta"), (ev.ca.find_urgency, "has_urgency")):
        y = df[col].to_numpy().astype(int)
        p = np.array([int(bool(fn(x))) for x in df.text])
        f = lambda a, b: f1_score(a, b, zero_division=0)
        P, R, F, _ = precision_recall_fscore_support(y, p, average="binary", zero_division=0)
        out[col] = {"precision": float(P), "recall": float(R), "f1": float(F), "f1_ci": boot(f, y, p, idx), "positives": int(y.sum())}
    return out


def image_block():
    out = {}
    for name, fn in (("tuning set", "results.json"), ("held-out set", "results_heldout.json")):
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            continue
        im = json.load(open(path))["images"]
        rows = {}
        k_all = n_all = 0
        for cat, e in im["by_category"].items():
            if e.get("passed") is None:
                continue
            k, n = int(e["passed"]), int(e["n"])
            k_all += k; n_all += n
            rows[cat] = {"passed": k, "n": n, "ci": wilson(k, n)}
        rows["ALL"] = {"passed": k_all, "n": n_all, "ci": wilson(k_all, n_all)}
        w = im.get("watermark_detection")
        if w:
            rows["watermark recall"] = {"passed": w["flagged"], "n": w["watermarked"], "ci": wilson(w["flagged"], w["watermarked"])}
            rows["watermark false alarms"] = {"passed": w["false_flags"], "n": w["unwatermarked"], "ci": wilson(w["false_flags"], w["unwatermarked"])}
        a = im.get("ai_detection")
        if a and a["ai_images"]:
            rows["AI recall"] = {"passed": a["ai_flagged"], "n": a["ai_images"], "ci": wilson(a["ai_flagged"], a["ai_images"])}
        if a:
            rows["AI false alarms"] = {"passed": a["real_flagged"], "n": a["real_images"], "ci": wilson(a["real_flagged"], a["real_images"])}
        out[name] = rows
    return out


def f(x):
    return f"{100 * x:.1f}%"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def write(res):
    L = ["# Statistics: confidence intervals and significance", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python eval/stats.py`. 95% intervals: bootstrap "
         f"({B:,} resamples of the ads, seed 0) for copy, Wilson for image rates.", ""]
    for split, blk in res["copy"].items():
        L += [f"## Ad copy sentiment, {split} ({blk['rows']} ads)", "",
              "| Model | Accuracy [95% CI] | Macro P | Macro R | Macro F1 [95% CI] |", "|---|---|---|---|---|"]
        for n, m in blk["models"].items():
            L.append(f"| {n} | {f(m['accuracy'])} {ci(m['accuracy_ci'])} | {m['macro_precision']:.2f} | "
                     f"{m['macro_recall']:.2f} | {m['macro_f1']:.2f} [{m['macro_f1_ci'][0]:.2f}, {m['macro_f1_ci'][1]:.2f}] |")
        L += ["", "Per-class precision / recall / F1 [95% CI of F1]:", "",
              "| Model | Negative | Neutral | Positive |", "|---|---|---|---|"]
        for n, m in blk["models"].items():
            cells = [f"{c['precision']:.2f} / {c['recall']:.2f} / {c['f1']:.2f} [{c['f1_ci'][0]:.2f}, {c['f1_ci'][1]:.2f}]"
                     for c in (m["per_class"][lab] for lab in LABELS)]
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        L += ["", "Exact McNemar tests (ads where only one model is right):", ""]
        for k, v in blk["mcnemar"].items():
            L.append(f"- {k}: {v['only_first_correct']} vs {v['only_second_correct']}, p = {v['p_value']:.2g}")
        L.append("")
    d = res["detectors"]
    L += ["## Copy detectors (90 held-out ads)", "", "| Detector | Positives | Precision | Recall | F1 [95% CI] |", "|---|---|---|---|---|"]
    for col, name in (("has_cta", "Call to action"), ("has_urgency", "Urgency")):
        x = d[col]
        L.append(f"| {name} | {x['positives']} | {x['precision']:.2f} | {x['recall']:.2f} | {x['f1']:.2f} [{x['f1_ci'][0]:.2f}, {x['f1_ci'][1]:.2f}] |")
    for name, rows in res["images"].items():
        L += ["", f"## Images, {name}", "", "| Measure | k / n | Rate [95% Wilson CI] |", "|---|---|---|"]
        for k, v in rows.items():
            L.append(f"| {k} | {v['passed']} / {v['n']} | {f(v['passed'] / v['n'])} {ci(v['ci'])} |")
    with open(os.path.join(HERE, "stats.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(HERE, "stats.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\n".join(L))


def main():
    all_df = pd.read_csv(os.path.join(HERE, "copy_test_set.csv"))
    test = all_df[all_df.split == "test"].reset_index(drop=True)
    preds_all = predictions(all_df)
    is_test = (all_df.split == "test").to_numpy()
    preds_test = {n: p[is_test] for n, p in preds_all.items()}
    res = {"copy": {}}
    res["copy"]["held-out test half"], idx = copy_block(test, preds_test)
    res["copy"]["all 180 ads (supplementary; the dev half was used to choose models and tune rules)"], _ = copy_block(all_df, preds_all)
    res["detectors"] = detector_block(test, idx)
    res["images"] = image_block()
    write(res)


if __name__ == "__main__":
    main()
