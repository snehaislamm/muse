"""Robustness of the position-bias conclusions to the simulator's assumptions.

The main result (IPW beats the naive ranker; the naive offline metric says the
opposite; IPS agrees with the truth) could be an artefact of the one
examination curve the simulator happens to use. This script regenerates the
logged impressions under different hidden curves and exploration budgets,
reruns the identical study code (muse_ml.position_study), and reports whether
each conclusion still holds.

Scenarios include one where the true curve is *not* in the power-law family
the propensity model fits (misspecification).

    python experiments/position_robustness.py      # needs `make data` first, ~3 min
"""

from __future__ import annotations

import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from muse_ml import datagen, features, position_study  # noqa: E402

ART, DATA = ROOT / "artifacts", ROOT / "data"


def power(b, a=2.0):
    return lambda k: (1.0 + np.asarray(k, dtype=float) / a) ** -b


SCENARIOS = {
    "weak bias (b=0.4)": (power(0.4), 0.10),
    "default (b=0.8)": (power(0.8), 0.10),
    "strong bias (b=1.4)": (power(1.4), 0.10),
    "misspecified: exponential decay": (lambda k: np.exp(-np.asarray(k, dtype=float) / 5.0), 0.10),
    "little exploration (2% randomized)": (power(0.8), 0.02),
}


def read_f32(path):
    with open(path, "rb") as f:
        rows, cols = struct.unpack("<II", f.read(8))
        return np.frombuffer(f.read(), dtype="<f4").reshape(rows, cols)


def main():
    world = datagen.load_world(DATA)
    z = np.load(DATA / "world.npz")
    saves_train = z["saves_train"]
    emb, visual = read_f32(ART / "emb.f32"), read_f32(ART / "visual.f32")
    pins = json.load(open(ART / "pins.json"))
    clusters = {int(k): v for k, v in json.load(open(ART / "user_clusters.json")).items()}
    topic_ids = np.array([p["topic_id"] for p in pins])
    counts = np.bincount(saves_train[:, 2], minlength=len(pins)).astype(float)
    static = features.pin_static(
        counts, np.array([p["created_day"] for p in pins]), np.array([p["spam_score"] for p in pins])
    )
    by_user = defaultdict(list)
    for u, _board, p, d in saves_train:
        by_user[int(u)].append((int(p), int(d)))
    states = {
        u: features.UserState(
            [r[0] for r in by_user.get(u, [])],
            [r[1] for r in by_user.get(u, [])],
            clusters.get(u, []),
            emb,
            visual,
            topic_ids,
        )
        for u in range(len(world.users))
    }
    eval_users = np.random.default_rng(3).random(len(world.users)) < 0.2
    train_list = [tuple(r) for r in saves_train]

    results = {}
    for name, (curve, rand_frac) in SCENARIOS.items():
        print(f"== {name}", flush=True)
        rng = np.random.default_rng(8)  # same candidate sets in every scenario
        imps = datagen.generate_impressions(world, rng, train_list, randomized_frac=rand_frac, exam_fn=curve)
        X = position_study.featurize_impressions(imps, states, emb, visual, topic_ids, static)
        tr = ~eval_users[imps[:, 0]]
        _, m = position_study.run(X, imps, tr, world, curve)
        ag = m["estimator_agreement"]
        results[name] = {
            "propensity_max_abs_error": m["propensity"]["max_abs_error"],
            "true_lift_ipw_over_naive": ag["true_lift_ipw_over_naive"],
            "naive_predicted_lift": ag["naive_expected_saves"]["predicted_lift_ipw_over_naive"],
            "ips_predicted_lift": ag["ips_expected_saves"]["predicted_lift_ipw_over_naive"],
            "naive_tau": ag["naive_expected_saves"]["kendall_tau_vs_truth"],
            "ips_tau": ag["ips_expected_saves"]["kendall_tau_vs_truth"],
            "oracle_vs_estimated_true_saves": [
                m["ranking_eval"]["mlp_ipw_oracle_propensity"]["true_expected_saves"],
                m["ranking_eval"]["mlp_ipw"]["true_expected_saves"],
            ],
        }
        print(json.dumps(results[name]), flush=True)

    out = ROOT / "experiments" / "results" / "position_robustness.json"
    out.write_text(json.dumps(results, indent=2))
    print(
        "\n| scenario | propensity max err | true lift (IPW vs naive) | naive predicts | IPS predicts "
        "| τ naive | τ IPS |\n|---|---|---|---|---|---|---|"
    )
    for name, r in results.items():
        print(
            f"| {name} | {r['propensity_max_abs_error']:.3f} | {r['true_lift_ipw_over_naive']:+.1%} "
            f"| {r['naive_predicted_lift']:+.1%} | {r['ips_predicted_lift']:+.1%} "
            f"| {r['naive_tau']:.2f} | {r['ips_tau']:.2f} |"
        )


if __name__ == "__main__":
    main()
