"""
10_image_model_training.py
--------------------------
Trains an image engagement model on the Instagram influencer dataset - the
trained, validated counterpart to the heuristic in 04_image_analysis.py.
Run 08_instagram_audit.py first: there's no point fitting a model to data
that fails the authenticity checks.

    python 10_image_model_training.py                    # train + evaluate + save
    python 10_image_model_training.py --download-images  # if the CSV has image URLs

What it learns
  Within each account, posts are ranked by engagement rate
  (likes [+ comments] / followers). The model predicts whether an image
  lands in that creator's TOP third or BOTTOM third. Ranking within an
  account removes follower count, niche and posting habits, so what's
  left to learn from is the image itself. It does NOT predict absolute likes.

How it's evaluated
  Frozen CLIP image embeddings + the visual features from 04 feed a
  logistic regression. GroupKFold by account: every fold's test accounts
  were never seen in training, so the model can't score well just by
  recognising a creator. Out-of-fold results are compared against:
    - chance (AUC 0.5)
    - the current heuristic (exact same math as 04_image_analysis.score_image)
    - visual features only
    - permuted labels (sanity check - should land near 0.5)
    - a random, NOT account-grouped split (reference only, shows leakage)
  If the trained model doesn't beat the heuristic, the report says so.

Reads  data/raw/<config.IMAGE_ENGAGEMENT["raw_file"]> + the images it points to
Writes results/image_engagement/{report.md, metrics.json, oof_predictions.csv}
       models/image_engagement.joblib
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

CFG = config.IMAGE_ENGAGEMENT
COLS = config.INSTAGRAM_AUDIT["columns"]
SEED = config.SEED
VIS_KEYS = ["brightness", "contrast", "saturation", "sharpness", "width", "height", "aspect_ratio"]
C_GRID = [0.001, 0.01, 0.1, 1.0]


def _load_module_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Module names starting with a digit can't be imported normally.
image_analysis = _load_module_by_path("image_analysis", os.path.join(config.BASE_DIR, "04_image_analysis.py"))


# ── Loading (same column parsing as 08_instagram_audit.py) ────────────
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


def to_time(s):
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.dropna().median() > 1e11 else "s"
        return pd.to_datetime(s, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(s, errors="coerce", utc=True)


def load_posts(args) -> tuple[pd.DataFrame, dict, list]:
    path = os.path.join(config.RAW_DIR, CFG["raw_file"])
    if not os.path.exists(path):
        sys.exit(f"Input file not found: {path}\nPut the Instagram CSV there (or set INSTAGRAM_RAW_FILE).")
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")

    lookup = {norm(c): c for c in df.columns}

    def find(cands, override):
        if override:
            if override not in df.columns:
                sys.exit(f"Column '{override}' not in the CSV. Columns: {list(df.columns)}")
            return override
        return next((lookup[c] for c in cands if c in lookup), None)

    cols = {
        "likes": find(COLS["likes"], args.likes_col),
        "comments": find(COLS["comments"], None),
        "followers": find(COLS["followers"], args.followers_col),
        "timestamp": find(COLS["timestamp"], None),
        "image": find(COLS["image"], args.image_col),
        "account": find(CFG["account_columns"], args.account_col),
    }
    missing = [r for r in ("likes", "image", "account") if cols[r] is None]
    if missing:
        sys.exit(f"Could not find column(s) for: {', '.join(missing)}.\n"
                 f"CSV columns: {list(df.columns)}\n"
                 f"Pass them explicitly, e.g. --account-col owner_username --image-col image_path")

    notes = []
    posts = pd.DataFrame({
        "account": df[cols["account"]].astype(str).str.strip(),
        "image_ref": df[cols["image"]].astype(str).str.strip(),
        "engagement": to_num(df[cols["likes"]]),
    })
    if cols["comments"]:
        posts["engagement"] = posts["engagement"] + to_num(df[cols["comments"]]).fillna(0)
    if cols["followers"]:
        posts["followers"] = to_num(df[cols["followers"]])
    else:
        notes.append("No followers column - engagement is ranked on likes alone. That's still a "
                     "within-account comparison, but follower growth over time isn't corrected for.")

    ok = posts["engagement"].ge(0) & ~posts["image_ref"].isin(["", "nan", "None"])
    if cols["followers"]:
        ok &= posts["followers"].gt(0)
    if (~ok).sum():
        notes.append(f"Dropped {(~ok).sum()} rows with missing/invalid likes, followers or image.")
    posts = posts[ok]

    if cols["timestamp"]:
        ts = to_time(df.loc[posts.index, cols["timestamp"]])
        if ts.notna().any():
            cutoff = ts.max() - pd.Timedelta(days=CFG["min_post_age_days"])
            fresh = ts > cutoff
            if fresh.sum():
                notes.append(f"Dropped {fresh.sum()} posts from the last {CFG['min_post_age_days']} days "
                             f"of the dataset - they were still collecting likes when scraped.")
            posts = posts[~fresh]
    else:
        notes.append("No timestamp column - can't drop posts that were too new to have settled.")

    return posts.reset_index(drop=True), cols, notes


def filter_accounts(posts: pd.DataFrame) -> pd.DataFrame:
    counts = posts.groupby("account")["account"].transform("size")
    return posts[counts >= CFG["min_posts_per_account"]].reset_index(drop=True)


# ── Images ─────────────────────────────────────────────────────────────
def resolve_image(ref: str, download: bool) -> str | None:
    if ref.startswith(("http://", "https://")):
        local = os.path.join(CFG["image_dir"], "downloaded", hashlib.sha1(ref.encode()).hexdigest()[:20] + ".jpg")
        if os.path.exists(local):
            return local
        if not download:
            return None
        import requests
        try:
            r = requests.get(ref, timeout=15)
            r.raise_for_status()
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, "wb") as f:
                f.write(r.content)
            Image.open(local).verify()
            return local
        except Exception:  # noqa: BLE001 - expired CDN links etc. are just skipped
            if os.path.exists(local):
                os.remove(local)
            return None
    for p in (ref, os.path.join(CFG["image_dir"], ref), os.path.join(config.RAW_DIR, ref)):
        if os.path.isfile(p):
            return p
    return None


def resolve_images(posts: pd.DataFrame, download: bool, notes: list) -> pd.DataFrame:
    refs = posts["image_ref"].unique()
    print(f"Resolving {len(refs)} images{' (downloading URLs)' if download else ''}...")
    mapping = {}
    for i, ref in enumerate(refs, 1):
        mapping[ref] = resolve_image(ref, download)
        if download and i % 200 == 0:
            print(f"  {i}/{len(refs)}")
    posts = posts.assign(image_path=posts["image_ref"].map(mapping))
    unresolved = posts["image_path"].isna()
    if unresolved.all():
        is_url = posts["image_ref"].str.startswith(("http://", "https://")).mean() > 0.5
        hint = ("The image column holds URLs - re-run with --download-images. Instagram CDN links "
                "expire, so if downloads fail you'll need the dataset's image files instead."
                if is_url else f"Put the image files in {CFG['image_dir']} (or set INSTAGRAM_IMAGE_DIR).")
        sys.exit(f"None of the {len(posts)} images could be found.\n{hint}")
    if unresolved.sum():
        notes.append(f"Dropped {unresolved.sum()} posts whose image couldn't be found or downloaded.")
    return posts[~unresolved].reset_index(drop=True)


# ── CLIP embeddings + visual features (cached) ────────────────────────
def _image_features_tensor(out):
    # transformers returns a tensor in v4 and a ModelOutput in newer versions
    return out if hasattr(out, "norm") else out.pooler_output


def embed(paths: list) -> tuple[dict, np.ndarray, float]:
    """Returns ({path: (L2-normalised CLIP embedding, visual feature vector)},
    normalised text embeddings of 04's prompts, CLIP logit scale)."""
    cache_path = CFG["embedding_cache"]
    cache, txt, scale = {}, None, None
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if str(z["model"]) == CFG["clip_model"]:
            cache = {p: (e, v) for p, e, v in zip(z["paths"], z["emb"], z["vis"])}
            txt, scale = z["txt"], float(z["scale"])

    todo = [p for p in dict.fromkeys(paths) if p not in cache]
    if todo or txt is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        print(f"Loading {CFG['clip_model']}...")
        model = CLIPModel.from_pretrained(CFG["clip_model"]).eval()
        processor = CLIPProcessor.from_pretrained(CFG["clip_model"])
        with torch.no_grad():
            prompts = image_analysis.ClipScorer.POSITIVE_PROMPTS + image_analysis.ClipScorer.NEGATIVE_PROMPTS
            t = _image_features_tensor(model.get_text_features(**processor(text=prompts, return_tensors="pt", padding=True)))
            txt = (t / t.norm(dim=-1, keepdim=True)).numpy()
            scale = float(model.logit_scale.exp())

            print(f"Embedding {len(todo)} new images ({len(cache)} cached)...")
            bs = CFG["batch_size"]
            for start in range(0, len(todo), bs):
                batch, images = [], []
                for p in todo[start:start + bs]:
                    try:
                        images.append(Image.open(p).convert("RGB"))
                        batch.append(p)
                    except Exception:  # noqa: BLE001 - corrupt file, skip it
                        pass
                if not images:
                    continue
                e = _image_features_tensor(model.get_image_features(**processor(images=images, return_tensors="pt")))
                e = (e / e.norm(dim=-1, keepdim=True)).numpy()
                for p, img, emb in zip(batch, images, e):
                    feats = image_analysis.extract_visual_features(img)
                    cache[p] = (emb, np.array([feats[k] for k in VIS_KEYS], dtype=float))
                done = min(start + bs, len(todo))
                if done % (bs * 10) == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)}")

        keys = list(cache)
        np.savez(cache_path, model=CFG["clip_model"], paths=np.array(keys),
                 emb=np.stack([cache[k][0] for k in keys]), vis=np.stack([cache[k][1] for k in keys]),
                 txt=txt, scale=scale)
    return cache, txt, scale


def heuristic_scores(emb: np.ndarray, vis: np.ndarray, txt: np.ndarray, scale: float) -> np.ndarray:
    """Exactly what 04_image_analysis.score_image() returns as 'score', computed
    from the cached embeddings instead of a second CLIP pass."""
    logits = scale * emb @ txt.T
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    n_pos = len(image_analysis.ClipScorer.POSITIVE_PROMPTS)
    pos, neg = probs[:, :n_pos].sum(1), probs[:, n_pos:].sum(1)
    clip_score = 100 * pos / (pos + neg + 1e-9)
    feature_score = np.array([image_analysis.score_visual_features(dict(zip(VIS_KEYS, row)))["feature_score"]
                              for row in vis])
    return 0.5 * clip_score + 0.5 * feature_score


def visual_matrix(vis: np.ndarray) -> np.ndarray:
    """Model-friendly version of the 04 visual features (logs for skewed ones)."""
    d = dict(zip(VIS_KEYS, vis.T))
    return np.column_stack([d["brightness"], d["contrast"], d["saturation"], np.log1p(d["sharpness"]),
                            np.log(d["aspect_ratio"]), np.log(d["width"] * d["height"])])


# ── Labels ─────────────────────────────────────────────────────────────
def add_labels(posts: pd.DataFrame) -> pd.DataFrame:
    rate = np.log1p(posts["engagement"])
    if "followers" in posts:
        rate = rate - np.log1p(posts["followers"])
    pct = rate.groupby(posts["account"]).rank(pct=True)
    q = CFG["label_quantile"]
    label = np.where(pct > 1 - q, 1.0, np.where(pct <= q, 0.0, np.nan))
    return posts.assign(log_rate=rate, pct_in_account=pct, label=label)


# ── Models + evaluation ───────────────────────────────────────────────
def make_model():
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    return GridSearchCV(pipe, {"logisticregression__C": C_GRID}, scoring="roc_auc", cv=GroupKFold(n_splits=3))


def cross_val_predict(X, y, groups, splitter) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold P(top third) for every post. Trains on labeled (top/bottom)
    posts only; predicts all test posts so the middle third counts for Spearman."""
    oof, fold = np.full(len(X), np.nan), np.full(len(X), -1)
    labeled = ~np.isnan(y)
    for k, (tr, te) in enumerate(splitter.split(X, groups=groups)):
        tr = tr[labeled[tr]]
        model = make_model()
        model.fit(X[tr], y[tr].astype(int), groups=groups[tr])
        oof[te], fold[te] = model.predict_proba(X[te])[:, 1], k
    return oof, fold


def within_account_spearman(score, pct, groups) -> float:
    rs = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 5 and np.ptp(score[m]) > 0:
            rs.append(spearmanr(score[m], pct[m]).correlation)
    return float(np.nanmean(rs)) if rs else float("nan")


def bootstrap_auc(y, score, groups, baseline=None) -> dict:
    """95% CI for AUC (and AUC minus baseline AUC) by resampling ACCOUNTS,
    since posts from one account aren't independent."""
    rng = np.random.default_rng(SEED)
    accounts = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in accounts}
    aucs, diffs = [], []
    for _ in range(CFG["bootstrap_reps"]):
        idx = np.concatenate([idx_by[g] for g in rng.choice(accounts, len(accounts))])
        if len(np.unique(y[idx])) < 2:
            continue
        a = roc_auc_score(y[idx], score[idx])
        aucs.append(a)
        if baseline is not None:
            diffs.append(a - roc_auc_score(y[idx], baseline[idx]))
    out = {"auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}
    if baseline is not None:
        out["delta_auc_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    return out


def summarize(name, score, y, pct, groups, fold=None, is_proba=True, baseline=None) -> dict:
    lab = ~np.isnan(y)
    yl, sl, gl = y[lab].astype(int), score[lab], groups[lab]
    row = {
        "method": name,
        "auc": float(roc_auc_score(yl, sl)),
        "accuracy": float(accuracy_score(yl, sl >= 0.5)) if is_proba else None,
        "within_account_spearman": within_account_spearman(score, pct, groups),
        **bootstrap_auc(yl, sl, gl, baseline[lab] if baseline is not None else None),
    }
    if fold is not None:
        per_fold = [roc_auc_score(yl[fold[lab] == k], sl[fold[lab] == k])
                    for k in np.unique(fold[lab]) if len(np.unique(yl[fold[lab] == k])) == 2]
        row["auc_fold_mean"], row["auc_fold_std"] = float(np.mean(per_fold)), float(np.std(per_fold))
    return row


def verdict(model_row: dict) -> str:
    lo, hi = model_row["auc_ci"]
    d_lo, d_hi = model_row["delta_auc_ci"]
    if lo <= 0.5:
        return ("The trained model is NOT distinguishable from chance on unseen accounts "
                f"(AUC 95% CI {lo:.3f}-{hi:.3f} includes 0.5). Don't show it as a prediction; "
                "keep the heuristic labelled as a heuristic.")
    if d_lo > 0:
        return (f"The trained model beats chance (AUC 95% CI {lo:.3f}-{hi:.3f}) and beats the current "
                f"heuristic (AUC difference 95% CI {d_lo:+.3f} to {d_hi:+.3f}) on accounts it never saw.")
    if d_hi < 0:
        return (f"The trained model beats chance (AUC 95% CI {lo:.3f}-{hi:.3f}) but is WORSE than the "
                f"current heuristic (AUC difference CI {d_lo:+.3f} to {d_hi:+.3f}).")
    return (f"The trained model beats chance (AUC 95% CI {lo:.3f}-{hi:.3f}), but the difference from the "
            f"heuristic isn't statistically clear (AUC difference CI {d_lo:+.3f} to {d_hi:+.3f}). "
            "More accounts would tighten this.")


# ── Reporting ──────────────────────────────────────────────────────────
def write_report(rows, posts, cols, notes, final_c, verdict_text):
    os.makedirs(CFG["out_dir"], exist_ok=True)
    n_lab = int(posts["label"].notna().sum())

    def fmt(v, spec=".3f"):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)

    lines = [
        "# Image engagement model - evaluation report", "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by 10_image_model_training.py_", "",
        "## Verdict", "", verdict_text, "",
        "## Data", "",
        f"- {len(posts)} posts from {posts['account'].nunique()} accounts "
        f"({n_lab} labeled top/bottom third, {len(posts) - n_lab} middle third used for Spearman only)",
        f"- Columns used: " + ", ".join(f"{k} = `{v}`" for k, v in cols.items() if v),
        f"- Target: within-account rank of log(engagement{' / followers' if cols['followers'] else ''})",
        *[f"- {n}" for n in notes], "",
        "## Results (out-of-fold, accounts never seen in training)", "",
        "| Method | AUC | AUC 95% CI | AUC per fold | Accuracy | Within-account Spearman |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        fold = f"{fmt(r['auc_fold_mean'])} ± {fmt(r['auc_fold_std'])}" if "auc_fold_mean" in r else "-"
        ci = f"{fmt(r['auc_ci'][0])}-{fmt(r['auc_ci'][1])}"
        lines.append(f"| {r['method']} | {fmt(r['auc'])} | {ci} | {fold} | {fmt(r['accuracy'])} | "
                     f"{fmt(r['within_account_spearman'])} |")
    lines += [
        "", "- **AUC**: chance of ranking a random top-third post above a random bottom-third post (0.5 = chance).",
        "- **Within-account Spearman**: does the score order each creator's posts correctly (0 = no signal).",
        "- CIs come from resampling accounts, since posts from one account aren't independent.",
        "- The permuted-label row should be ~0.5. If it isn't, something leaks.",
        "- The random-split row is NOT a valid estimate: the same accounts appear in train and test. "
        "It's shown only to illustrate how much an ungrouped split would flatter the model.",
        "", "## Final model", "",
        f"Logistic regression (C = {final_c}) on {CFG['clip_model']} image embeddings + 6 visual features, "
        f"refit on all labeled posts. Saved to `{os.path.relpath(CFG['model_path'], config.BASE_DIR)}`.",
        "", "It predicts P(this image would be in its creator's top third), i.e. relative performance "
        "for a given account - not absolute likes, and not validated for accounts unlike the ones in this dataset.",
    ]
    with open(os.path.join(CFG["out_dir"], "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(CFG["out_dir"], "metrics.json"), "w") as f:
        json.dump({"results": rows, "verdict": verdict_text, "n_posts": len(posts),
                   "n_accounts": int(posts["account"].nunique()), "n_labeled": n_lab}, f, indent=2)


# ── Inference ──────────────────────────────────────────────────────────
class ImageEngagementPredictor:
    """Loads the saved model for use elsewhere (e.g. the dashboard):

        m = _load_module_by_path("image_model", "10_image_model_training.py")
        predictor = m.ImageEngagementPredictor()
        predictor.predict(Image.open("post.jpg"))
        # -> {"probability_top_third": 0.64, "metrics": {...}}
    """

    def __init__(self, model_path: str = CFG["model_path"]):
        import joblib
        from transformers import CLIPModel, CLIPProcessor

        self.bundle = joblib.load(model_path)
        self.clip = CLIPModel.from_pretrained(self.bundle["clip_model"]).eval()
        self.processor = CLIPProcessor.from_pretrained(self.bundle["clip_model"])

    def predict(self, image: Image.Image) -> dict:
        import torch

        image = image.convert("RGB")
        with torch.no_grad():
            e = _image_features_tensor(self.clip.get_image_features(**self.processor(images=image, return_tensors="pt")))
        e = (e / e.norm(dim=-1, keepdim=True)).numpy()
        feats = image_analysis.extract_visual_features(image)
        vis = visual_matrix(np.array([[feats[k] for k in VIS_KEYS]], dtype=float))
        p = float(self.bundle["model"].predict_proba(np.hstack([e, vis]))[0, 1])
        return {"probability_top_third": p, "metrics": self.bundle["metrics"]}


# ── Main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--download-images", action="store_true", help="download image URLs into the image folder")
    ap.add_argument("--account-col", help="column identifying the account (overrides auto-detection)")
    ap.add_argument("--image-col", help="column with the image path or URL")
    ap.add_argument("--likes-col", help="column with the like count")
    ap.add_argument("--followers-col", help="column with the follower count")
    args = ap.parse_args()

    posts, cols, notes = load_posts(args)
    posts = resolve_images(filter_accounts(posts), args.download_images, notes)

    cache, txt, scale = embed(posts["image_path"].tolist())
    unreadable = ~posts["image_path"].isin(cache.keys())
    if unreadable.sum():
        notes.append(f"Dropped {unreadable.sum()} posts whose image file couldn't be opened.")
    posts = add_labels(filter_accounts(posts[~unreadable]))

    n_acc = posts["account"].nunique()
    if n_acc < max(CFG["min_accounts"], CFG["n_splits"]):
        sys.exit(f"Only {n_acc} accounts have >= {CFG['min_posts_per_account']} usable posts; need at least "
                 f"{max(CFG['min_accounts'], CFG['n_splits'])} for an account-grouped evaluation to mean anything.")
    print(f"\n{len(posts)} posts, {n_acc} accounts, {int(posts['label'].notna().sum())} labeled")

    emb = np.stack([cache[p][0] for p in posts["image_path"]])
    vis = np.stack([cache[p][1] for p in posts["image_path"]])
    X_vis = visual_matrix(vis)
    X = np.hstack([emb, X_vis])
    y = posts["label"].to_numpy()
    pct = posts["pct_in_account"].to_numpy()
    groups = posts["account"].to_numpy()
    grouped = GroupKFold(n_splits=CFG["n_splits"])

    print("Cross-validating (grouped by account)...")
    heur = heuristic_scores(emb, vis, txt, scale)
    p_model, f_model = cross_val_predict(X, y, groups, grouped)
    p_vis, f_vis = cross_val_predict(X_vis, y, groups, grouped)

    rng = np.random.default_rng(SEED)
    y_perm = posts.groupby("account")["label"].transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()
    p_perm, f_perm = cross_val_predict(X, y_perm, groups, grouped)

    p_leak, f_leak = cross_val_predict(X, y, groups, KFold(n_splits=CFG["n_splits"], shuffle=True, random_state=SEED))

    rows = [
        summarize("Trained: CLIP embedding + visual features", p_model, y, pct, groups, f_model, baseline=heur),
        summarize("Current heuristic (04 score_image)", heur, y, pct, groups, is_proba=False),
        summarize("Trained: visual features only", p_vis, y, pct, groups, f_vis),
        summarize("Sanity: permuted labels", p_perm, y_perm, pct, groups, f_perm),
        summarize("Reference only: random split (leaky)", p_leak, y, pct, groups, f_leak),
    ]
    verdict_text = verdict(rows[0])

    print("Fitting final model on all labeled posts...")
    import joblib

    lab = ~np.isnan(y)
    final = make_model().fit(X[lab], y[lab].astype(int), groups=groups[lab])
    final_c = final.best_params_["logisticregression__C"]
    os.makedirs(os.path.dirname(CFG["model_path"]), exist_ok=True)
    joblib.dump({"model": final.best_estimator_, "clip_model": CFG["clip_model"], "vis_keys": VIS_KEYS,
                 "label_quantile": CFG["label_quantile"], "metrics": rows[0],
                 "trained_on": {"posts": len(posts), "accounts": n_acc, "file": CFG["raw_file"]}},
                CFG["model_path"])

    os.makedirs(CFG["out_dir"], exist_ok=True)
    posts.drop(columns=["image_ref"]).assign(
        heuristic=heur, pred_model=p_model, pred_visual_only=p_vis, fold=f_model,
    ).to_csv(os.path.join(CFG["out_dir"], "oof_predictions.csv"), index=False)
    write_report(rows, posts, cols, notes, final_c, verdict_text)

    print()
    for r in rows:
        print(f"  {r['method']:<44} AUC {r['auc']:.3f}  [{r['auc_ci'][0]:.3f}-{r['auc_ci'][1]:.3f}]  "
              f"Spearman {r['within_account_spearman']:+.3f}")
    print(f"\n{verdict_text}")
    print(f"\nReport -> {os.path.relpath(os.path.join(CFG['out_dir'], 'report.md'), config.BASE_DIR)}")
    print(f"Model  -> {os.path.relpath(CFG['model_path'], config.BASE_DIR)}")


if __name__ == "__main__":
    main()
