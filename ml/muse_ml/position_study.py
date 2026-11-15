"""The position-bias study as one reusable function, so the main pipeline and
the robustness sweep (experiments/position_robustness.py) run identical code.

Given logged impressions (with slots), features and the hidden world, it:
  1. estimates propensities on the randomized slice of training users,
  2. trains three rankers: naive, IPW with estimated propensities, and IPW with
     the true propensities (an oracle reference),
  3. scores 7 rankers on held-out users' logged sessions with the naive, IPS and
     ground-truth estimators, and reports how well each estimator agrees with
     the truth.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import kendalltau

from muse_ml import datagen, debias, features, ranker

SESSION = 25
ESTIMATORS = ("naive_expected_saves", "ips_expected_saves", "naive_ndcg10", "ips_ndcg10")


def featurize_impressions(imps, states, emb, visual, topic_ids, static) -> np.ndarray:
    X = np.zeros((len(imps), len(features.FEATURES)))
    for u, st in states.items():
        rows = np.where(imps[:, 0] == u)[0]
        if len(rows):
            X[rows] = features.featurize(st, imps[rows, 1], emb, visual, topic_ids, static)
    return X


def utility(model: ranker.MultiTaskMLP, X: np.ndarray) -> np.ndarray:
    p = model.predict(X)
    return p[:, 1] + 0.3 * p[:, 0] - 2.0 * p[:, 2]


def run(
    X: np.ndarray,
    imps: np.ndarray,
    train_rows: np.ndarray,
    world: datagen.World,
    exam_true: Callable[[np.ndarray], np.ndarray],
) -> tuple[dict, dict]:
    """Returns (models, metrics). `train_rows` is a boolean mask over impressions."""
    Y = imps[:, 2:5].astype(float)
    pos, randomized = imps[:, 5], imps[:, 6].astype(bool)
    tr = train_rows

    # 1. propensities from randomized traffic of training users
    rmask = randomized & tr
    a_hat, b_hat, emp = debias.estimate_propensity(imps[rmask, 2], pos[rmask])
    slots = np.arange(SESSION)
    est_curve = debias.power_law(slots, a_hat, b_hat)
    metrics: dict = {
        "propensity": {
            "a_hat": a_hat,
            "b_hat": b_hat,
            "randomized_impressions": int(rmask.sum()),
            "estimated": est_curve.round(4).tolist(),
            "empirical_ratio": np.round(emp, 4).tolist(),
            "true": exam_true(slots).round(4).tolist(),
            "max_abs_error": float(np.abs(est_curve - exam_true(slots)).max()),
        }
    }

    # 2. rankers
    models = {
        "naive": ranker.MultiTaskMLP(X.shape[1]).fit(X[tr], Y[tr]),
        "ipw": ranker.MultiTaskMLP(X.shape[1]).fit(X[tr], debias.ipw_targets(Y[tr], pos[tr], a_hat, b_hat)),
        "ipw_oracle": ranker.MultiTaskMLP(X.shape[1]).fit(
            X[tr], debias.ipw_targets_from_propensity(Y[tr], exam_true(pos[tr]))
        ),
    }

    # 3. counterfactual evaluation on held-out users' non-randomized sessions
    session_id = np.arange(len(imps)) // SESSION
    eval_sessions = np.unique(session_id[~tr & ~randomized])
    sessions, idx = [], []
    for sid in eval_sessions:
        rows = np.arange(sid * SESSION, (sid + 1) * SESSION)
        user = int(imps[rows[0], 0])
        true_save = datagen.truth_probs(world, user, imps[rows, 1])["save"]
        sessions.append({"save": Y[rows, 1], "pos": pos[rows], "true_save": true_save})
        idx.append(rows)

    rng = np.random.default_rng(0)
    rankings = {
        "logging_policy": [-pos[r].astype(float) for r in idx],
        "random": [rng.random(len(r)) for r in idx],
        "popularity": [X[r, 4] for r in idx],
        "embedding_similarity": [X[r, 0] for r in idx],
        "mlp_naive": [utility(models["naive"], X[r]) for r in idx],
        "mlp_ipw": [utility(models["ipw"], X[r]) for r in idx],
        "mlp_ipw_oracle_propensity": [utility(models["ipw_oracle"], X[r]) for r in idx],
    }

    def prop_hat(p):
        return debias.power_law(p, a_hat, b_hat)

    evals = {
        name: debias.counterfactual_metrics(sessions, scores, prop_hat, exam_true) for name, scores in rankings.items()
    }
    metrics["ranking_eval"] = evals
    metrics["ranking_eval_sessions"] = len(sessions)

    names = list(evals)
    truth = np.array([evals[n]["true_expected_saves"] for n in names])
    agreement = {}
    for est in ESTIMATORS:
        values = np.array([evals[n][est] for n in names])
        agreement[est] = {
            "kendall_tau_vs_truth": float(kendalltau(values, truth).statistic),
            "predicted_lift_ipw_over_naive": evals["mlp_ipw"][est] / evals["mlp_naive"][est] - 1,
        }
    agreement["true_lift_ipw_over_naive"] = (
        evals["mlp_ipw"]["true_expected_saves"] / evals["mlp_naive"]["true_expected_saves"] - 1
    )
    metrics["estimator_agreement"] = agreement

    ev = ~tr
    metrics["ranker_auc_logged_labels"] = {
        name: {f"auc_{h}": ranker.auc(Y[ev, i], models[name].predict(X[ev])[:, i]) for i, h in enumerate(ranker.HEADS)}
        for name in ("naive", "ipw")
    }
    return models, metrics
