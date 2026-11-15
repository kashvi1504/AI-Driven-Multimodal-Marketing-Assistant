"""
eval/compare_rewriters.py
-------------------------
Compares rewriting models on the same weak ads and writes
eval/rewrite_comparison.md.

    python eval/compare_rewriters.py
    python eval/compare_rewriters.py --models google/flan-t5-base Qwen/Qwen2.5-1.5B-Instruct

Ads: the positive and neutral ads in the held-out test split of
copy_test_set.csv that score below 70 (the ones the app would rewrite),
each tried with no offer details. Every candidate goes through the same
grounding check the app uses.
"""
import argparse
import importlib.util
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config  # noqa: E402
import copy_analysis as ca  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["google/flan-t5-base", "Qwen/Qwen2.5-1.5B-Instruct"])
    ap.add_argument("--candidates", type=int, default=8)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("mt", os.path.join(ROOT, "02_model_training.py"))
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    sentiment = mt.SentimentPredictor(config.COPY_SENTIMENT_MODEL)

    df = pd.read_csv(os.path.join(HERE, "copy_test_set.csv"))
    df = df[(df.split == "test") & (df.label != "negative")].copy()
    preds = sentiment.predict(df.text.tolist())
    df["label_pred"] = [p["label"] for p in preds]
    df["score"] = [ca.predict_engagement(ca.analyze_text_features(t), p)["engagement_score"] for t, p in zip(df.text, preds)]
    df = df[df.score < 70]

    device = ca.compute_device_name()
    summary, examples = [], {}
    for name in args.models:
        cls = ca._Seq2SeqRewriter if "t5" in name.lower() else ca._ChatRewriter
        rewriter = cls(name, device)
        t = time.time()
        rows = []
        for text, label, score in zip(df.text, df.label_pred, df.score):
            gen = rewriter.generate_batch([rewriter.prompt(text, "")], temperature=0.7, top_p=0.9,
                                          num_return_sequences=args.candidates)[0]
            res = ca._choose_rewrite(text, "", ca.sentiment_prior_from_label(label), score, gen)
            copied = sum(1 for g in gen if ca._clean(g).lower().strip(".!") == text.lower().strip(".!"))
            rows.append({"text": text, "rewrite": res["refined_text"], "gain": res["score_change"],
                         "template": res["method"].startswith("safe template"),
                         "cta": bool(ca.find_cta(res["refined_text"])),
                         "rejected": len(res["rejected_candidates"]), "copied": copied,
                         "unsupported": len(res["unsupported_claims"])})
        r = pd.DataFrame(rows)
        summary.append({"model": name.split("/")[-1], "ads": len(r), "seconds_per_ad": (time.time() - t) / len(r),
                        "model_rewrite_used": 1 - r.template.mean(), "has_cta": r.cta.mean(), "avg_gain": r.gain.mean(),
                        "copied_input": r.copied.sum() / (len(r) * args.candidates),
                        "rejected": int(r.rejected.sum()), "unsupported_in_final": int(r.unsupported.sum())})
        examples[name.split("/")[-1]] = r
        print(summary[-1], flush=True)
        del rewriter

    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    L = ["# Rewriter comparison", "",
         f"{len(df)} weak ads from the held-out test split (positive and neutral, score < 70), no offer details, "
         f"{args.candidates} candidates each, device: {device}.", "",
         "| Model | Model's own rewrite used | Candidates that just copied the ad | Final rewrite has a call to action | "
         "Avg score gain | Candidates rejected for invented claims | Invented claims in final rewrites | Time per ad |",
         "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        L.append(f"| {s['model']} | {pct(s['model_rewrite_used'])} | {pct(s['copied_input'])} | {pct(s['has_cta'])} | "
                 f"+{s['avg_gain']:.1f} | {s['rejected']} | {s['unsupported_in_final']} | {s['seconds_per_ad']:.1f} s |")
    names = list(examples)
    L += ["", "When the model's own rewrite isn't used, the app falls back to a safe template (your ad plus a neutral call to action).", "",
          "## Examples", "", "| Original | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for i in range(min(12, len(df))):
        L.append(f"| {df.text.iloc[i]} | " + " | ".join(examples[n].rewrite.iloc[i] for n in names) + " |")
    with open(os.path.join(HERE, "rewrite_comparison.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
