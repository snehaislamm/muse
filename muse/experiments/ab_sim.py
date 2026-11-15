"""Online A/B test simulator against the running Go service.

Simulated Pinners (driven by the hidden ground-truth model from datagen.py)
open their home feed once per "day", scroll with position bias, and click /
save / hide according to their true preferences. Saves and hides are POSTed
back, so each variant's feed adapts online exactly as it would in production.

Users are randomised 50/50 into control and treatment by a salted hash.
Primary metric: saves per user. Guardrails: hides per user, p95 latency.
Subgroup: users who prefer an under-represented skin-tone range (3 or 4).

    python experiments/ab_sim.py --control baseline --treatment full --users 1200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from muse_ml.datagen import TONE_TOPICS, examine_prob, load_world, truth_probs  # noqa: E402


def call(base, path, body=None):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode() if body else None, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def arm_of(user: int, salt: str) -> str:
    return "treatment" if int(hashlib.sha256(f"{salt}:{user}".encode()).hexdigest(), 16) % 2 else "control"


def simulate_user(base, w, uid, variant, days, n, seed, extra=""):
    rng = np.random.default_rng(seed)
    out = {"saves": 0, "hides": 0, "clicks": 0, "impressions": 0, "tone_saves": 0, "fresh_saves": 0, "lat": []}
    for _ in range(days):
        r = call(base, f"/v1/feed?user={uid}&n={n}&variant={variant}{extra}")
        out["lat"].append(r["debug"]["stage_ms"]["total"])
        items = r["items"]
        if not items:
            continue
        pins = np.array([it["id"] for it in items])
        pr = truth_probs(w, uid, pins)
        examine = rng.random(len(pins)) < examine_prob(np.arange(len(pins)))  # same position bias as the logs
        for i, it in enumerate(items):
            if not examine[i]:
                continue
            out["impressions"] += 1
            out["clicks"] += int(rng.random() < pr["click"][i])
            if rng.random() < pr["save"][i]:
                out["saves"] += 1
                out["tone_saves"] += int(it["tone"] > 0)
                out["fresh_saves"] += int(it["created_day"] >= 79)
                call(base, "/v1/events", {"user": uid, "pin": int(it["id"]), "type": "save"})
            elif rng.random() < pr["hide"][i]:
                out["hides"] += 1
                call(base, "/v1/events", {"user": uid, "pin": int(it["id"]), "type": "hide"})
    return out


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t = (b.mean() - a.mean()) / se if se else 0.0
    p = math.erfc(abs(t) / math.sqrt(2))  # normal approx, n is large
    return t, p


def bootstrap_lift(a, b, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    lifts = [(rng.choice(b, len(b)).mean() / max(rng.choice(a, len(a)).mean(), 1e-9) - 1) for _ in range(iters)]
    return float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--control", default="baseline")
    ap.add_argument("--treatment", default="full")
    ap.add_argument("--users", type=int, default=1200)
    ap.add_argument("--offset", type=int, default=0, help="first user id (use disjoint ranges per experiment)")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--treatment-params", default="", help="e.g. 'lambda=0.95&cap=3' (re-ranking overrides)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    w = load_world(ROOT / "data")
    salt = f"{args.control}-vs-{args.treatment}"
    users = list(range(args.offset, args.offset + args.users))
    arms = {u: arm_of(u, salt) for u in users}

    def run(u):
        if arms[u] == "treatment":
            extra = "&" + args.treatment_params if args.treatment_params else ""
            return u, simulate_user(args.url, w, u, args.treatment, args.days, args.n, seed=u, extra=extra)
        return u, simulate_user(args.url, w, u, args.control, args.days, args.n, seed=u)

    with ThreadPoolExecutor(4) as ex:
        results = dict(ex.map(run, users))

    topic_name = {p["topic_id"]: p["topic"] for p in w.pins}
    under = {
        u
        for u in users
        if w.users[u]["tone_pref"] in (3, 4)
        and any(topic_name[it["topic_id"]] in TONE_TOPICS for it in w.users[u]["interests"])
    }

    report = {
        "control": args.control,
        "treatment": args.treatment,
        "treatment_params": args.treatment_params,
        "users": len(users),
        "days": args.days,
        "arm_sizes": {a: sum(1 for x in arms.values() if x == a) for a in ("control", "treatment")},
    }
    for metric in ["saves", "hides", "clicks", "fresh_saves"]:
        a = [results[u][metric] for u in users if arms[u] == "control"]
        b = [results[u][metric] for u in users if arms[u] == "treatment"]
        t, p = welch(a, b)
        lo, hi = bootstrap_lift(a, b)
        report[metric] = {
            "control_mean": float(np.mean(a)),
            "treatment_mean": float(np.mean(b)),
            "lift": float(np.mean(b) / max(np.mean(a), 1e-9) - 1),
            "ci95": [lo, hi],
            "p_value": p,
        }
    a = [results[u]["tone_saves"] for u in under if arms[u] == "control"]
    b = [results[u]["tone_saves"] for u in under if arms[u] == "treatment"]
    if len(a) > 5 and len(b) > 5:
        lo, hi = bootstrap_lift(a, b)
        report["underrepresented_tone_users_tone_saves"] = {
            "n": len(a) + len(b),
            "control_mean": float(np.mean(a)),
            "treatment_mean": float(np.mean(b)),
            "lift": float(np.mean(b) / max(np.mean(a), 1e-9) - 1),
            "ci95": [lo, hi],
            "p_value": welch(a, b)[1],
        }
    for arm in ("control", "treatment"):
        lat = [x for u in users if arms[u] == arm for x in results[u]["lat"]]
        report[f"server_latency_ms_{arm}"] = {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
        }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
