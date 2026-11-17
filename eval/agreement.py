"""
eval/agreement.py
-----------------
Inter-annotator agreement between the original labels (eval/copy_test_set.csv)
and an independent second annotator (eval/annotation/second_annotator.csv,
see eval/annotation/INSTRUCTIONS.md). Writes eval/agreement.md.

    python eval/agreement.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
B = 10_000


def boot_kappa(a, b, rng):
    n = len(a)
    vals = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(set(a[i])) > 1 or len(set(b[i])) > 1:
            vals.append(cohen_kappa_score(a[i], b[i]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    gold = pd.read_csv(os.path.join(HERE, "copy_test_set.csv"))
    second = pd.read_csv(os.path.join(HERE, "annotation", "second_annotator.csv"), dtype=str)
    second = second.dropna(subset=["label"])
    second = second[second.label.str.strip() != ""]
    if second.empty:
        sys.exit("second_annotator.csv has no labels yet - see eval/annotation/INSTRUCTIONS.md")
    second["label"] = second.label.str.strip().str.lower()
    for c in ("has_cta", "has_urgency"):
        second[c] = second[c].astype(float).astype(int)
    second["id"] = second.id.astype(int)
    m = gold.merge(second, on="id", suffixes=("", "_2"))
    rng = np.random.default_rng(0)
    L = ["# Inter-annotator agreement", "", f"{len(m)} ads labelled by both annotators.", "",
         "| Column | Subset | Raw agreement | Cohen's kappa [95% bootstrap CI] |", "|---|---|---|---|"]
    for col in ("label", "has_cta", "has_urgency"):
        for name, sub in (("all", m), ("held-out test half", m[m.split == "test"])):
            a, b = sub[col].to_numpy(), sub[col + "_2"].to_numpy()
            if len(a) == 0:
                continue
            k = cohen_kappa_score(a, b)
            lo, hi = boot_kappa(a, b, rng)
            L.append(f"| {col} | {name} ({len(a)}) | {np.mean(a == b):.1%} | {k:.2f} [{lo:.2f}, {hi:.2f}] |")
    dis = m[m.label != m.label_2]
    if len(dis):
        L += ["", "Tone disagreements:", ""]
        L += [f"- ({r.label} vs {r.label_2}) {r.text}" for r in dis.itertuples()]
    with open(os.path.join(HERE, "agreement.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
