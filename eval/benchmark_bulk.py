"""
eval/benchmark_bulk.py
----------------------
Times Bulk Analysis (the same _run_bulk the web app uses) on CSVs of
different sizes, with and without rewrites, and writes eval/bulk_benchmark.md.

    python eval/benchmark_bulk.py
    python eval/benchmark_bulk.py --sizes 50 100 1000 --rewrite-sizes 50 100

Rows are built by pairing sentences from eval/copy_test_set.csv, so every
row is different (repeats would be served from cache and look faster).
"""
import argparse
import importlib.util
import itertools
import os
import platform
import random
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def make_rows(n, seed=0):
    texts = pd.read_csv(os.path.join(HERE, "copy_test_set.csv")).text.tolist()
    pairs = list(itertools.permutations(texts, 2))
    random.Random(seed).shuffle(pairs)
    singles = [(t,) for t in texts]
    pool = singles + pairs
    return pd.DataFrame({"text": [" ".join(p) for p in pool[:n]]})


def run(app, df, rewrite):
    job_id = f"bench-{len(df)}-{rewrite}-{time.time()}"
    app._jobs[job_id] = {"status": "running", "progress": 0.0, "created": time.time(), "rows": len(df)}
    t = time.time()
    app._run_bulk(job_id, df, "text", rewrite)
    job = app._jobs.pop(job_id)
    if job["status"] != "done":
        raise RuntimeError(job)
    return time.time() - t, job["result"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 1000, 5000])
    ap.add_argument("--rewrite-sizes", type=int, nargs="+", default=[50, 100])
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("web_app", os.path.join(ROOT, "03_web_app.py"))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app.predictor()  # load the sentiment model before timing
    run(app, make_rows(5), True)  # load the rewriter before timing

    rows = []
    for n in args.sizes:
        secs, _ = run(app, make_rows(n, seed=n), False)
        rows.append({"rows": n, "rewrites": "off", "seconds": secs, "rows_per_sec": n / secs})
        print(rows[-1], flush=True)
    for n in args.rewrite_sizes:
        secs, res = run(app, make_rows(n, seed=10_000 + n), True)
        out = pd.read_csv(__import__("io").StringIO(res["csv"]))
        changed = out[out.refined_text.astype(str) != out.text.astype(str)]
        rows.append({"rows": n, "rewrites": "on", "seconds": secs, "rows_per_sec": n / secs,
                     "rewritten": len(changed), "avg_gain": changed.heuristic_score_change.mean(),
                     "rejected_for_invented_claims": int(out.rewrites_rejected_for_invented_claims.sum())})
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    on = df[df.rewrites == "on"]
    # the largest run is the most realistic mix of rows that need rewriting
    per_rewrite_row = (on.seconds / on.rows).iloc[on.rows.argmax()] if len(on) else None
    lines = ["# Bulk Analysis benchmark", "",
             f"Machine: {platform.machine()} · {platform.platform()} · device: {app.ca.compute_device_name()}", "",
             "| Rows | Rewrites | Time | Rows per second | Rows rewritten | Avg score gain | Candidates rejected for invented claims |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        extra = (f"{r['rewritten']} | +{r['avg_gain']:.1f} | {r['rejected_for_invented_claims']}"
                 if r["rewrites"] == "on" else "- | - | -")
        lines.append(f"| {r['rows']:,} | {r['rewrites']} | {r['seconds']:.1f} s | {r['rows_per_sec']:.1f} | {extra} |")
    if per_rewrite_row:
        lines += ["", f"With rewrites on, each row costs about {per_rewrite_row:.2f} s, so 1,000 rows take about "
                      f"{per_rewrite_row * 1000 / 60:.0f} min and 5,000 rows about {per_rewrite_row * 5000 / 60:.0f} min."]
    with open(os.path.join(HERE, "bulk_benchmark.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
