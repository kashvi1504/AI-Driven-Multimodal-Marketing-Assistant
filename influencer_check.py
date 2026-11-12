"""
influencer_check.py
-------------------
Influencer Check (web app page): does an Instagram account's engagement
look organic, or are there signs of bought followers or likes?

Bought followers don't like or comment, so an account that bought them has
far fewer likes than its follower count suggests. Bought likes show up as
likes with almost no comments, or as single posts that spike far above the
account's normal level. This module turns those signals into a heuristic
verdict with reasons. It's a screening aid, not proof: niche, posting
frequency and recent virality all move these numbers too.

The "typical engagement" floor per account size lives in
config.INFLUENCER_CHECK, so it can be adjusted as benchmarks change.
Instagram doesn't allow scraping, so numbers are entered or uploaded.
"""

import math
import re
import statistics

from config import INFLUENCER_CHECK as CFG


def tier(followers: int) -> dict:
    for t in CFG["tiers"]:
        if followers < t["max_followers"]:
            return t
    return CFG["tiers"][-1]


# 1,000,000 or Indian 1,20,000 grouping; anything else ("870,31") is two numbers
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?|\d+(?:\.\d+)?")


def parse_posts(text: str) -> list:
    """'likes, comments' per line (comments optional) -> [(likes, comments|None)]."""
    posts = []
    for line in str(text or "").splitlines():
        nums = [float(n.replace(",", "")) for n in _NUMBER.findall(line)]
        if nums:
            posts.append((nums[0], nums[1] if len(nums) > 1 else None))
    return posts


def check_account(followers: float, avg_likes: "float | None" = None, avg_comments: "float | None" = None,
                  following: "float | None" = None, posts: "list | None" = None, name: str = "") -> dict:
    if not followers or followers <= 0:
        raise ValueError("Enter the follower count.")
    posts = posts or []
    likes_list = [p[0] for p in posts]
    comments_list = [p[1] for p in posts if p[1] is not None]
    if likes_list:
        avg_likes = statistics.fmean(likes_list)
    if comments_list:
        avg_comments = statistics.fmean(comments_list)
    if avg_likes is None:
        raise ValueError("Enter average likes, or paste likes per post.")
    avg_comments = avg_comments or 0.0

    t = tier(followers)
    like_rate = 100 * avg_likes / followers
    engagement = 100 * (avg_likes + avg_comments) / followers
    comments_per_100 = 100 * avg_comments / avg_likes if avg_likes else 0.0
    flags, notes, risk = [], [], 0

    floor, typical = t["min_rate"], t["typical_rate"]
    if engagement < floor / 2:
        risk += 3
        flags.append(f"Engagement {engagement:.2f}% is far below the {floor}% floor for {t['name']} accounts. "
                     "Most followers never interact - a classic sign of bought or inactive followers.")
    elif engagement < floor:
        risk += 1
        flags.append(f"Engagement {engagement:.2f}% is below the usual {floor}% floor for {t['name']} accounts.")
    elif engagement > CFG["suspicious_high_rate"]:
        risk += 1
        flags.append(f"Engagement {engagement:.1f}% is unusually high. Fine after a viral post, "
                     "otherwise check for bought likes or engagement pods.")
    else:
        notes.append(f"Engagement {engagement:.2f}% is healthy for a {t['name']} account "
                     f"(typical ~{typical}%, floor {floor}%).")

    if avg_comments and comments_per_100 < CFG["min_comments_per_100_likes"]:
        risk += 2
        flags.append(f"Only {comments_per_100:.2f} comments per 100 likes. Bought likes rarely come with comments.")
    elif avg_comments == 0:
        notes.append("No comment numbers given, so the likes-to-comments check was skipped.")
    else:
        notes.append(f"{comments_per_100:.1f} comments per 100 likes - a normal conversation level.")

    spikes = []
    if len(likes_list) >= 5:
        median = statistics.median(likes_list)
        spikes = [i + 1 for i, v in enumerate(likes_list) if median and v > CFG["spike_factor"] * median]
        if spikes:
            risk += 1
            flags.append(f"Post(s) {', '.join(map(str, spikes))} got over {CFG['spike_factor']}x the usual likes. "
                         "Check whether that was a viral post or a paid like boost.")
        cv = statistics.pstdev(likes_list) / statistics.fmean(likes_list) if statistics.fmean(likes_list) else 0
        notes.append(f"Likes vary by {100 * cv:.0f}% from post to post across {len(likes_list)} posts.")

    if following and followers < CFG["follow_back_max_followers"] and following > followers:
        risk += 1
        flags.append(f"Follows {int(following):,} accounts but has {int(followers):,} followers - "
                     "often a follow-for-follow growth tactic.")

    verdict = ("Likely inflated audience" if risk >= 3 else "Some warning signs" if risk >= 1 else "Looks organic")
    authenticity = max(0, 100 - 22 * risk)
    return {
        "name": name, "followers": followers, "following": following, "tier": t["name"],
        "avg_likes": avg_likes, "avg_comments": avg_comments, "posts": len(likes_list),
        "like_rate": like_rate, "engagement_rate": engagement, "comments_per_100_likes": comments_per_100,
        "typical_rate": typical, "min_rate": floor, "expected_likes": followers * floor / 100,
        "spike_posts": spikes, "risk_points": risk, "authenticity": authenticity, "verdict": verdict,
        "flags": flags, "notes": notes,
    }


def _num(value):
    try:
        v = float(str(value).replace(",", "").strip())
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def check_accounts_table(df) -> list:
    """One account per row. Needs followers and likes columns; comments,
    following and username are optional. Column names are matched loosely."""
    cols = {c: re.sub(r"[^a-z]", "", str(c).lower()) for c in df.columns}

    def find(*keys):
        return next((c for c, k in cols.items() if any(key in k for key in keys)), None)

    fcol, lcol = find("follower"), find("like")
    ccol, gcol, ncol = find("comment"), find("following"), find("user", "account", "handle", "name")
    if gcol == fcol:
        gcol = None
    if not fcol or not lcol:
        raise ValueError("The CSV needs a followers column and a likes column (average likes per post).")
    results = []
    for _, row in df.iterrows():
        try:
            results.append(check_account(_num(row[fcol]), _num(row[lcol]), _num(row[ccol]) if ccol else None,
                                         _num(row[gcol]) if gcol else None, name=str(row[ncol]) if ncol else ""))
        except ValueError as e:
            results.append({"name": str(row[ncol]) if ncol else "", "error": str(e)})
    return results
