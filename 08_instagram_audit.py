"""
08_instagram_audit.py
---------------------
Authenticity audit for the Instagram influencer dataset. Run BEFORE training:

    python 08_instagram_audit.py

Reads  data/raw/<config.INSTAGRAM_AUDIT["raw_file"]>
Writes results/instagram_audit/*.png and audit_report.md

Statistics can show data is *consistent with* real behaviour or *inconsistent*
(synthetic-looking). They can never prove authenticity; the provenance section
of the report must be filled in by hand.
"""
import os
import sys

import numpy as np
import pandas as pd

import config

try:
    from scipy import stats
except ImportError:
    print("scipy is required: pip install scipy (see Requirements.txt)")
    sys.exit(1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG = getattr(config, "INSTAGRAM_AUDIT", None)
if CFG is None:
    print("config.INSTAGRAM_AUDIT is missing - add the block from the task "
          "notes to config.py, then re-run.")
    sys.exit(1)

DEFAULT_TH = {
    "uniform_band": (0.90, 1.10), "uniform_band_warn": (0.75, 1.25),
    "min_skew": 1.0, "min_span": 2.0,
    "benford_mad_pass": 0.012, "benford_mad_fail": 0.015,   # Nigrini bands
    "corr_pass": 0.30, "corr_warn": 0.10,
    "ratio_over1_warn": 0.05, "ratio_over1_fail": 0.20,
    "eng_over1_warn": 0.02, "eng_over1_fail": 0.20,
    "likes_gt_f_warn": 0.20, "likes_gt_f_fail": 0.50,
    "future_fail": 0.01,
    "dup_caption_warn": 0.20, "dup_caption_fail": 0.50,
    "dup_image_warn": 0.05, "dup_image_fail": 0.30,
    "dup_id_fail": 0.001,
    "group_spread_fail": 1.10, "group_eta_warn": 0.01,
    "funnel_fail": 0.01, "derived_tol": 1e-4, "derived_match_pass": 0.95,
    "hour_flat_ratio": 1.25, "acct_small_n": 50, "min_accounts": 8,
}
TH = {**DEFAULT_TH, **CFG.get("thresholds", {})}
OUT = CFG["out_dir"]
MIN_ROWS = CFG.get("min_rows", 200)
os.makedirs(OUT, exist_ok=True)
RANK = {"PASS": 0, "SKIP": 0, "WARN": 1, "FAIL": 2}
RESULTS, KEY, NOTES = [], {}, []
CORR_CHECK = "Likes vs followers correlation"


def add(check, verdict, reason):
    RESULTS.append((check, verdict, reason))
    print(f"[{verdict:4}] {check}: {reason}")


def norm(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def to_num(s):
    """Numbers, '1,234', '12.5K', '1.2M' -> float; anything else -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype(str).str.strip().str.replace(",", "", regex=False).str.upper()
    mult = t.str[-1].map({"K": 1e3, "M": 1e6, "B": 1e9}).fillna(1.0)
    base = pd.to_numeric(t.where(mult == 1.0, t.str[:-1]), errors="coerce")
    return base * mult


def load():
    path = os.path.join(config.RAW_DIR, CFG["raw_file"])
    if not os.path.exists(path):
        found = [f for f in os.listdir(config.RAW_DIR) if f.lower().endswith(".csv")] \
            if os.path.isdir(config.RAW_DIR) else []
        return None, {}, (f"Input file not found: {path}. Put the Instagram CSV there "
                          f"(or set INSTAGRAM_RAW_FILE). CSVs in data/raw/: {found or 'none'}.")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:  # noqa: BLE001
        try:
            df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        except Exception:
            return None, {}, f"Could not read {path}: {e}"
    lookup = {norm(c): c for c in df.columns}
    cols = {role: next((lookup[c] for c in cands if c in lookup), None)
            for role, cands in CFG["columns"].items()}
    return df, cols, None


def loglog_hist(x, name, bins_n=30):
    x = x[x > 0]
    if len(x) < 10 or x.max() == x.min():
        return
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), bins_n)
    h, e = np.histogram(x, bins=bins, density=True)
    ctr, ok = np.sqrt(e[:-1] * e[1:]), h > 0
    plt.figure(figsize=(5, 4))
    plt.loglog(ctr[ok], h[ok], "o-")
    plt.xlabel(f"{name} (log)"); plt.ylabel("density (log)")
    plt.title(f"{name}: log-log histogram\n(real data: roughly straight, declining tail)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"loglog_{name}.png"), dpi=120)
    plt.close()


def span_of(x):
    x = x[x > 0]
    return float(np.log10(x.max() / x.min())) if len(x) > 1 else 0.0


# ── 1. Distribution shape ─────────────────────────────────────────
def check_distribution(d, df, cols):
    g = cols.get("group")
    for name in ("likes", "followers"):
        label = f"Distribution shape ({name})"
        if name not in d:
            add(label, "SKIP", f"no {name} column found")
            continue
        x = d[name].dropna()
        x = x[x >= 0]
        # followers are an account attribute: judge them per account, not per post
        acct_level = False
        if name == "followers" and g:
            per = d["followers"].groupby(df[g]).agg(["nunique", "first"])
            if len(per) <= TH["acct_small_n"] and (per["nunique"] <= 1).all():
                x, acct_level = per["first"].dropna(), True
                NOTES.append(f"'{cols['followers']}' is constant within each of {len(per)} "
                             f"'{g}' accounts, so followers were judged at account level "
                             f"(n={len(per)}), not per post.")
        if len(x) < (3 if acct_level else MIN_ROWS):
            add(label, "SKIP", f"only {len(x)} usable rows")
            continue
        span = span_of(x)
        KEY[f"{name}: span (orders of magnitude)"] = round(span, 2)
        KEY[f"{name}: distinct values"] = int(x.nunique())
        if acct_level:
            loglog_hist(x, name, bins_n=max(4, len(x) // 3))
            v = "WARN" if span < TH["min_span"] else "PASS"
            add(label, v, f"{len(x)} accounts, follower range spans {span:.1f} orders of magnitude"
                + (" - too narrow for an influencer pool (real ones usually span >= 2)"
                   if v == "WARN" else " - wide range"))
            continue
        loglog_hist(x, name)
        rng = x.max() - x.min()
        uni_sd = rng / np.sqrt(12) if rng > 0 else np.nan
        ratio = x.std() / uni_sd
        skew = float(stats.skew(x))
        mm = x.mean() / max(x.median(), 1e-9)
        KEY[f"{name}: SD ratio (obs/uniform)"] = round(float(ratio), 3)
        KEY[f"{name}: skewness"] = round(skew, 2)
        KEY[f"{name}: mean/median"] = round(float(mm), 2)
        lo, hi = TH["uniform_band"]
        wlo, whi = TH["uniform_band_warn"]
        txt = (f"SD ratio {ratio:.2f} (1.00 = uniform), skew {skew:.1f}, "
               f"mean/median {mm:.1f}, span {span:.1f} orders")
        if lo <= ratio <= hi and skew < TH["min_skew"]:
            add(label, "FAIL", txt + " - matches a uniform distribution")
        elif skew < TH["min_skew"] or (wlo <= ratio <= whi and mm < 1.3) or span < TH["min_span"]:
            add(label, "WARN", txt + " - not clearly heavy-tailed / range too narrow")
        else:
            add(label, "PASS", txt + " - heavy-tailed over a wide range")


# ── 2. Benford ────────────────────────────────────────────────────
def check_benford(d):
    exp = np.log10(1 + 1 / np.arange(1, 10))
    for name in ("likes", "comments", "followers", "reach", "impressions"):
        label = f"Benford ({name})"
        if name not in d:
            if name in ("likes", "comments", "followers"):
                add(label, "SKIP", f"no {name} column found")
            continue
        x = d[name].dropna()
        x = x[x >= 1]
        if len(x) < 100:
            add(label, "SKIP", f"only {len(x)} values >= 1 (need >= 100)")
            continue
        if x.nunique() < 100:
            add(label, "SKIP", f"only {x.nunique()} distinct values (account-level attribute; Benford not applicable)")
            continue
        span = span_of(x)
        dig = x.astype(np.int64).astype(str).str[0].astype(int)
        obs = np.array([(dig == k).sum() for k in range(1, 10)])
        chi, p = stats.chisquare(obs, exp * obs.sum())
        mad = float(np.abs(obs / obs.sum() - exp).mean())
        KEY[f"Benford {name}: chi2 / p / MAD"] = f"{chi:.1f} / {p:.3g} / {mad:.4f}"
        plt.figure(figsize=(5, 4))
        plt.bar(range(1, 10), obs / obs.sum(), alpha=.7, label="observed")
        plt.plot(range(1, 10), exp, "ro-", label="Benford")
        plt.xlabel("leading digit"); plt.ylabel("share"); plt.legend()
        plt.title(f"Benford: {name} (MAD={mad:.4f})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"benford_{name}.png"), dpi=120)
        plt.close()
        txt = f"MAD={mad:.4f}, chi2={chi:.1f}, p={p:.3g} (uninformative at large n), span={span:.1f} orders"
        if span < TH["min_span"]:
            add(label, "SKIP", txt + " - range too narrow for Benford to apply")
        elif mad > TH["benford_mad_fail"]:
            add(label, "FAIL", txt + " - nonconformity (Nigrini MAD > 0.015)")
        elif mad > TH["benford_mad_pass"]:
            add(label, "WARN", txt + " - marginal conformity")
        else:
            add(label, "PASS", txt + " - conforms (MAD within Nigrini's acceptable band)")


# ── 3. Internal consistency ───────────────────────────────────────
def corr_verdict(r):
    return "PASS" if r >= TH["corr_pass"] else "WARN" if r >= TH["corr_warn"] else "FAIL"


def check_consistency(d, df, cols):
    g = cols.get("group")
    if "likes" in d and "followers" in d:
        m = d[["likes", "followers"]].dropna()
        m = m[(m.likes >= 0) & (m.followers >= 0)]
        if len(m) >= MIN_ROWS and m.followers.nunique() > 1:
            r = float(stats.spearmanr(m.likes, m.followers)[0])
            rp = float(np.corrcoef(np.log1p(m.likes), np.log1p(m.followers))[0, 1])
            KEY["likes~followers (post level): Spearman / log-Pearson"] = f"{r:.2f} / {rp:.2f}"
            extra = ""
            if g:
                a = pd.DataFrame({"g": df[g], "l": d["likes"], "f": d["followers"]}).dropna() \
                    .groupby("g").mean()
                if len(a) >= TH["min_accounts"]:
                    ra, pa = stats.spearmanr(a.l, a.f)
                    KEY[f"likes~followers (account means, n={len(a)}): Spearman / p"] = f"{ra:.2f} / {pa:.3g}"
                    extra = f"; account-mean Spearman {ra:.2f} (n={len(a)}, p={pa:.2g})"
            plt.figure(figsize=(5, 4))
            plt.loglog(m.followers + 1, m.likes + 1, ".", alpha=.3)
            plt.xlabel("followers"); plt.ylabel("likes")
            plt.title(f"likes vs followers (Spearman {r:.2f})")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT, "likes_vs_followers.png"), dpi=120)
            plt.close()
            v = corr_verdict(r)
            add(CORR_CHECK, v, f"Spearman {r:.2f}, log-Pearson {rp:.2f}{extra}" + {
                "PASS": " - clearly positive", "WARN": " - weak",
                "FAIL": " - ~none/negative; real audiences scale with followers"}[v])
        else:
            add(CORR_CHECK, "SKIP", "too few rows or followers has no variation")
    else:
        add(CORR_CHECK, "SKIP", "likes or followers column missing")

    if "likes" in d and "reach" in d:
        m = d[["likes", "reach"]].dropna()
        if len(m) >= MIN_ROWS:
            r = float(stats.spearmanr(m.likes, m.reach)[0])
            KEY["likes~reach: Spearman"] = round(r, 2)
            v = corr_verdict(r)
            add("Likes vs reach correlation", v, f"Spearman {r:.2f}" + {
                "PASS": " - clearly positive", "WARN": " - weak",
                "FAIL": " - ~none; likes should scale with how many people saw the post"}[v])

    if "likes" in d and "comments" in d:
        m = d[["likes", "comments"]].dropna()
        m = m[m.likes > 0]
        if len(m) >= MIN_ROWS:
            ratio = m.comments / m.likes
            med, over = float(ratio.median()), float((ratio > 1).mean())
            KEY["comments/likes: median / share >1"] = f"{med:.3f} / {over:.1%}"
            txt = f"median comments/likes {med:.3f}, {over:.1%} of posts have more comments than likes"
            if med >= 1 or over > TH["ratio_over1_fail"]:
                add("Comments/likes ratio", "FAIL", txt + " - implausible")
            elif med > 0.5 or over > TH["ratio_over1_warn"]:
                add("Comments/likes ratio", "WARN", txt + " - unusually high")
            else:
                add("Comments/likes ratio", "PASS", txt + " - typical (well under 1)")
        else:
            add("Comments/likes ratio", "SKIP", "too few rows")
    else:
        add("Comments/likes ratio", "SKIP", "likes or comments column missing")

    if "likes" in d and "followers" in d:
        cm = d["comments"] if "comments" in d else 0
        er = ((d["likes"] + cm) / d["followers"].where(d["followers"] > 0)).dropna()
        er = er[np.isfinite(er)]
        zero_f = int(((d["followers"] <= 0) & (d["likes"] > 0)).sum())
        if len(er) >= MIN_ROWS:
            over, med = float((er > 1).mean()), float(er.median())
            KEY["engagement rate: median / share >100% / likes with 0 followers"] = f"{med:.2%} / {over:.1%} / {zero_f}"
            txt = f"median {med:.2%}, {over:.1%} above 100%, {zero_f} posts with likes but 0 followers"
            if over > TH["eng_over1_fail"] or med > 1:
                add("Engagement rate", "FAIL", txt + " - impossible values")
            elif over > TH["eng_over1_warn"] or zero_f > 0:
                add("Engagement rate", "WARN", txt + " - some impossible/extreme values")
            else:
                add("Engagement rate", "PASS", txt + " - plausible")
        else:
            add("Engagement rate", "SKIP", "too few rows with followers > 0")
    else:
        add("Engagement rate", "SKIP", "likes or followers column missing")


def check_derived(d):
    """Is the stored engagement_rate reproducible from the raw counts?"""
    if "eng_rate" not in d:
        add("Stored engagement_rate", "SKIP", "no engagement_rate column found")
        return
    inter = sum(d[k] for k in ("likes", "comments", "shares", "saves") if k in d)
    lc = sum(d[k] for k in ("likes", "comments") if k in d)
    best, share = None, 0.0
    for den in ("reach", "impressions", "followers"):
        if den not in d:
            continue
        dd = d[den].where(d[den] > 0)
        for nm, num in (("likes+comments+shares+saves", inter), ("likes+comments", lc)):
            diff = (num / dd - d["eng_rate"]).abs().dropna()
            if len(diff) >= MIN_ROWS:
                s = float((diff <= TH["derived_tol"]).mean())
                if s > share:
                    best, share = f"({nm})/{den}", s
    if best is None:
        add("Stored engagement_rate", "SKIP", "not enough columns to recompute it")
        return
    KEY["stored engagement_rate: best formula / match share"] = f"{best} / {share:.1%}"
    if share >= TH["derived_match_pass"]:
        add("Stored engagement_rate", "PASS",
            f"{share:.1%} of rows match {best} - internally consistent, but it is a derived "
            "field so it is NOT independent evidence of authenticity")
    else:
        add("Stored engagement_rate", "WARN",
            f"cannot be reproduced from raw counts (best match {share:.1%} via {best})")


# ── 4. Impossible / suspicious values ─────────────────────────────
def check_impossible(df, d, cols):
    n = len(df)
    neg = {k: int((d[k] < 0).sum()) for k in ("likes", "comments", "followers") if k in d}
    if neg:
        tot = sum(neg.values())
        add("Negative counts", "FAIL" if tot else "PASS",
            f"{tot} negative values {neg}" if tot else "no negative counts")
    else:
        add("Negative counts", "SKIP", "no count columns found")

    if "likes" in d and "followers" in d:
        m = d[["likes", "followers"]].dropna()
        share = float((m.likes > m.followers).mean()) if len(m) else 0.0
        KEY["likes > followers: share"] = f"{share:.1%}"
        v = "FAIL" if share > TH["likes_gt_f_fail"] else "WARN" if share > TH["likes_gt_f_warn"] else "PASS"
        add("Likes far above followers", v, f"{share:.1%} of posts have likes > followers")
    else:
        add("Likes far above followers", "SKIP", "likes or followers column missing")

    if "reach" in d or "impressions" in d:
        viol, parts = 0, []
        if "reach" in d and "impressions" in d:
            m = d[["reach", "impressions"]].dropna()
            s = float((m.impressions < m.reach).mean()) if len(m) else 0.0
            viol = max(viol, s); parts.append(f"impressions<reach {s:.1%}")
        if "reach" in d and "likes" in d:
            m = d[["reach", "likes"]].dropna()
            s = float((m.likes > m.reach).mean()) if len(m) else 0.0
            viol = max(viol, s); parts.append(f"likes>reach {s:.1%}")
        if parts:
            v = "FAIL" if viol > TH["funnel_fail"] else "WARN" if viol > 0 else "PASS"
            add("Funnel logic (impressions >= reach >= likes)", v, ", ".join(parts))

    if "timestamp" in d:
        ts = d["timestamp"]
        bad = int(ts.isna().sum())
        fut = int((ts > pd.Timestamp.now(tz="UTC")).sum())
        KEY["timestamps: future / unparseable"] = f"{fut} / {bad}"
        v = "FAIL" if fut / max(n, 1) > TH["future_fail"] else "WARN" if fut or bad / max(n, 1) > 0.05 else "PASS"
        add("Timestamps", v, f"{fut} in the future, {bad} unparseable of {n}")
        hrs = ts.dropna().dt.hour
        if hrs.nunique() > 1 and len(hrs) >= MIN_ROWS:
            cnt = hrs.value_counts().reindex(range(24), fill_value=0)
            chi, p = stats.chisquare(cnt.values)
            ratio = float(cnt.max() / max(cnt.min(), 1))
            KEY["posting hour: max/min share / uniform-test p"] = f"{ratio:.2f} / {p:.3g}"
            txt = f"busiest/quietest hour ratio {ratio:.2f}, chi-square vs uniform p={p:.3g}"
            if p > 0.05 or ratio < TH["hour_flat_ratio"]:
                add("Posting-hour pattern", "WARN", txt + " - flat; real accounts post in daily peaks")
            else:
                add("Posting-hour pattern", "PASS", txt + " - clustered like real posting")
        else:
            add("Posting-hour pattern", "SKIP", "timestamps have no time-of-day information")
    else:
        add("Timestamps", "SKIP", "no timestamp column found")

    for role, warn, fail in (("caption", TH["dup_caption_warn"], TH["dup_caption_fail"]),
                             ("image", TH["dup_image_warn"], TH["dup_image_fail"])):
        c = cols.get(role)
        if not c:
            add(f"Duplicate {role}s", "SKIP", f"no {role} column found")
            continue
        s = df[c].dropna().astype(str).str.strip()
        s = s[s != ""]
        dup = float(s.duplicated(keep="first").mean()) if len(s) else 0.0
        KEY[f"duplicate {role}s"] = f"{dup:.1%}"
        v = "FAIL" if dup > fail else "WARN" if dup > warn else "PASS"
        add(f"Duplicate {role}s", v, f"{dup:.1%} of non-empty {role} values are repeats")

    c = cols.get("id")
    if c:
        dup = float(df[c].dropna().duplicated().mean())
        add("Duplicate post IDs", "FAIL" if dup > TH["dup_id_fail"] else "PASS",
            f"{dup:.2%} of post IDs are repeats")


# ── 5. Groups ─────────────────────────────────────────────────────
def check_groups(df, d, cols):
    g = cols.get("group")
    if not g or "likes" not in d:
        add("Uniformity across groups", "SKIP",
            "no account/category column" if not g else "no likes column")
        return
    m = pd.DataFrame({"g": df[g], "y": np.log1p(d["likes"].clip(lower=0))}).dropna()
    sizes = m.groupby("g").size()
    m = m[m.g.isin(sizes[sizes >= 3].index)]
    k = m.g.nunique()
    if k < 3 or len(m) < MIN_ROWS:
        add("Uniformity across groups", "SKIP", f"only {k} groups with >= 3 rows")
        return
    F, p = stats.f_oneway(*[x.y.values for _, x in m.groupby("g")])
    gm = d["likes"].groupby(df[g]).mean()
    eta2 = 1 - ((m.y - m.groupby("g").y.transform("mean")) ** 2).sum() / ((m.y - m.y.mean()) ** 2).sum()
    spread = float(gm.max() / max(gm.min(), 1e-9))
    KEY[f"group '{g}': n / ANOVA p / eta^2 / max-min mean ratio"] = f"{k} / {p:.3g} / {eta2:.3f} / {spread:.2f}"
    txt = f"{k} groups by '{g}', ANOVA p={p:.3g}, between-group variance share {eta2:.1%}, max/min mean {spread:.2f}"
    if p > 0.05 and spread < TH["group_spread_fail"]:
        v, why = "FAIL", " - group means are suspiciously identical"
    elif p > 0.05 or eta2 < TH["group_eta_warn"]:
        v, why = "WARN", " - little group structure"
    else:
        v, why = "PASS", " - groups differ as real accounts/categories do"
    add("Uniformity across groups", v, txt + why)


# ── Conclusion & report ───────────────────────────────────────────
def conclude():
    scored = [r for r in RESULTS if r[1] != "SKIP"]
    fails = sum(r[1] == "FAIL" for r in scored)
    warns = sum(r[1] == "WARN" for r in scored)
    key_fail = any(c == CORR_CHECK and v == "FAIL" for c, v, _ in RESULTS)
    if len(scored) < 4:
        return "INCONCLUSIVE", "Fewer than 4 checks could be evaluated (missing columns/rows)."
    if fails >= 2:
        return "LIKELY SYNTHETIC", f"{fails} checks FAILED, {warns} WARN."
    if key_fail and warns >= 1:
        return ("QUESTIONABLE - LEANING SYNTHETIC",
                f"{fails} FAIL, {warns} WARN. Likes do not scale with followers (a core property of "
                "real data) and other checks are also flagged - do not train until explained.")
    if fails == 1 or warns >= 3:
        return "QUESTIONABLE", f"{fails} FAIL, {warns} WARN - inspect the flagged checks before training."
    return "LIKELY REAL", (f"{fails} FAIL, {warns} WARN. Consistent with real data, "
                           "but statistics cannot prove authenticity - verify provenance.")


def write_report(status_msg=None, n_rows=None, cols=None):
    L = ["# Instagram dataset authenticity audit", ""]
    if status_msg:
        L += [f"**Audit could not be run:** {status_msg}", "",
              "No results are reported because no data was analysed.", ""]
    else:
        concl, why = conclude()
        L += [f"Rows analysed: {n_rows}", "", "## Overall conclusion", "",
              f"**{concl}** - {why}", "", "## Checks", "",
              "| Check | Verdict | Reason |", "|---|---|---|"]
        L += [f"| {c} | {v} | {r} |" for c, v, r in RESULTS]
        L += ["", "## Key numbers", ""] + [f"- {k}: {v}" for k, v in KEY.items()]
        L += ["", "## Column detection", ""]
        L += [f"- {role}: `{c}`" if c else f"- {role}: **not found**" for role, c in cols.items()]
        if NOTES:
            L += ["", "## Missing data / limitations", ""] + [f"- {x}" for x in NOTES]
        L += ["", "Plots: `loglog_*.png`, `benford_*.png`, `likes_vs_followers.png` in this folder."]
    L += ["", "## Provenance (fill in manually)", "",
          "- Source / where the dataset came from: TODO",
          "- License / terms of use: TODO",
          "- Collection date / period: TODO",
          "- Collection method (API, scrape, purchased, shared by influencers): TODO",
          "", "_Passing statistical checks does not prove authenticity; provenance must be documented above._"]
    with open(os.path.join(OUT, "audit_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nReport written to {os.path.join(OUT, 'audit_report.md')}")


def main():
    df, cols, err = load()
    if err:
        print(err)
        write_report(status_msg=err)
        return
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    for role, c in cols.items():
        print(f"  {role:11} -> {c if c else 'NOT FOUND'}")
    if not any(cols.get(k) for k in ("likes", "comments", "followers")):
        msg = ("No likes/comments/followers column detected. Columns found: "
               f"{list(df.columns)}. Add the right names to config.INSTAGRAM_AUDIT['columns'].")
        print(msg)
        write_report(status_msg=msg)
        return
    d = pd.DataFrame(index=df.index)
    for k in ("likes", "comments", "followers", "reach", "impressions", "shares", "saves", "eng_rate"):
        if cols.get(k):
            d[k] = to_num(df[cols[k]])
            bad = int(d[k].isna().sum())
            if bad:
                NOTES.append(f"{bad} non-numeric/missing values in '{cols[k]}' were ignored.")
    if cols.get("timestamp"):
        t = df[cols["timestamp"]]
        if pd.api.types.is_numeric_dtype(t):
            unit = "ms" if t.dropna().median() > 1e11 else "s"
            d["timestamp"] = pd.to_datetime(t, unit=unit, errors="coerce", utc=True)
        else:
            d["timestamp"] = pd.to_datetime(t, errors="coerce", utc=True)
    for role in ("likes", "comments", "followers", "timestamp", "caption", "image", "group"):
        if not cols.get(role):
            NOTES.append(f"No {role} column found - checks that need it were skipped.")
    if not cols.get("image"):
        NOTES.append("No image/image-path column: this file is tabular metrics only, so it cannot "
                     "support image-based engagement scoring on its own.")
    if not cols.get("caption"):
        NOTES.append("No caption text column (a caption-length number is not a caption).")
    if len(df) < MIN_ROWS:
        NOTES.append(f"Only {len(df)} rows; results are unreliable (min {MIN_ROWS}).")
    print()
    for fn, a in ((check_distribution, (d, df, cols)), (check_benford, (d,)),
                  (check_consistency, (d, df, cols)), (check_derived, (d,)),
                  (check_impossible, (df, d, cols)), (check_groups, (df, d, cols))):
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            add(fn.__name__, "SKIP", f"check crashed and was skipped: {e}")
            NOTES.append(f"{fn.__name__} failed with: {e}")
    concl, why = conclude()
    print(f"\nOVERALL: {concl} - {why}")
    write_report(n_rows=len(df), cols=cols)


if __name__ == "__main__":
    main()