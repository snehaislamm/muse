"""Position-bias correction (unbiased learning-to-rank, Joachims et al. 2017).

Logged labels are confounded by position: a pin shown in slot 0 is examined
about 8x more often than one in slot 24, so a model trained on raw labels learns
"whatever the old policy put on top is good". Under the examination hypothesis
P(action) = P(examined | pos) * P(action | examined, pin), so dividing each
label by the examination propensity gives an unbiased estimate of relevance.

1. `estimate_propensity` fits p(pos) = (1 + pos/a)^-b from the small randomized
   slice of traffic, where relevance is independent of position by construction.
2. `ipw_targets` divides labels by the (clipped) propensity. With a sigmoid +
   BCE head, the gradient is (s - y/p), whose expectation equals the gradient
   under the true relevance labels.
3. `counterfactual_metrics` estimates the expected saves a ranking would get,
   sum_i rel_i * p(new position of i), using (a) raw logged saves as rel (naive),
   (b) saves / p(logged position) as rel (IPS, unbiased), and (c) the hidden
   ground-truth save probability. nDCG variants are reported too.
"""

from __future__ import annotations

import numpy as np


def power_law(pos, a, b):
    return (1.0 + np.asarray(pos, dtype=float) / a) ** -b


def estimate_propensity(actions: np.ndarray, positions: np.ndarray, n_pos: int = 25):
    """Max-likelihood fit of p(pos) on randomized traffic (grid search).

    actions: 0/1 click indicator per randomized impression; positions: slot index.
    Returns (a, b, per-position empirical ratios normalised to slot 0).
    """
    best, best_ll = (1.0, 1.0), -np.inf
    for a in np.linspace(0.5, 6.0, 56):
        for b in np.linspace(0.2, 2.0, 91):
            p = power_law(positions, a, b)
            r = actions.mean() / p.mean()  # mean relevance (position-independent)
            q = np.clip(r * p, 1e-6, 1 - 1e-6)
            ll = (actions * np.log(q) + (1 - actions) * np.log(1 - q)).sum()
            if ll > best_ll:
                best, best_ll = (float(a), float(b)), ll
    ctr = np.array([actions[positions == k].mean() for k in range(n_pos)])
    return best[0], best[1], ctr / ctr[0]


def ipw_targets(Y: np.ndarray, positions: np.ndarray, a: float, b: float, clip: float = 0.1) -> np.ndarray:
    return ipw_targets_from_propensity(Y, power_law(positions, a, b), clip)


def ipw_targets_from_propensity(Y: np.ndarray, propensity: np.ndarray, clip: float = 0.1) -> np.ndarray:
    return Y / np.maximum(propensity, clip)[:, None]


def _dcg(rel, k):
    rel = rel[:k]
    return float((rel / np.log2(np.arange(2, len(rel) + 2))).sum())


def counterfactual_metrics(sessions, scores, prop_hat, prop_true, k=10):
    """sessions: list of dicts with logged 'save', 'pos', and hidden 'true_save' arrays.
    scores: list of score arrays (one per session) for the ranking being evaluated."""
    naive, ips, truth, exp_saves, naive_es, ips_es = [], [], [], [], [], []
    for s, sc in zip(sessions, scores, strict=True):
        order = np.argsort(-sc)
        for vals, out in (
            (s["save"].astype(float), naive),
            (s["save"] / prop_hat(s["pos"]), ips),
            (s["true_save"], truth),
        ):
            ideal = _dcg(np.sort(vals)[::-1], k)
            if ideal > 0:
                out.append(_dcg(vals[order], k) / ideal)
        # expected saves if this order were shown: sum_i rel_i * p(new position of i)
        new_pos = np.empty(len(order), dtype=int)
        new_pos[order] = np.arange(len(order))
        exam_new = prop_true(new_pos)
        exp_saves.append(float((exam_new * s["true_save"]).sum()))  # ground truth
        naive_es.append(float((exam_new * s["save"]).sum()))  # logged labels as relevance
        ips_es.append(float((exam_new * s["save"] / prop_hat(s["pos"])).sum()))  # IPS-corrected relevance
    return {
        "naive_ndcg10": float(np.mean(naive)),
        "ips_ndcg10": float(np.mean(ips)),
        "true_ndcg10": float(np.mean(truth)),
        "naive_expected_saves": float(np.mean(naive_es)),
        "ips_expected_saves": float(np.mean(ips_es)),
        "true_expected_saves": float(np.mean(exp_saves)),
    }
