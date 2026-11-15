import numpy as np
from muse_ml import debias


def test_propensity_recovered_from_randomized_traffic():
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 25, size=60000)
    rel = 0.3
    clicks = (rng.random(len(pos)) < rel * debias.power_law(pos, 2.0, 0.8)).astype(int)
    a, b, emp = debias.estimate_propensity(clicks, pos)
    est = debias.power_law(np.arange(25), a, b)
    assert np.abs(est - debias.power_law(np.arange(25), 2.0, 0.8)).max() < 0.03
    assert emp[0] == 1.0


def test_ipw_targets_are_unbiased_for_relevance():
    rng = np.random.default_rng(1)
    n = 400000
    pos = rng.integers(0, 25, size=n)
    relevance = 0.2
    observed = (rng.random(n) < debias.power_law(pos, 2.0, 0.8)) & (rng.random(n) < relevance)
    y = debias.ipw_targets(observed[:, None].astype(float), pos, 2.0, 0.8)[:, 0]
    assert abs(observed.mean() - relevance) > 0.05  # raw labels are biased
    assert abs(y.mean() - relevance) < 0.005  # IPW labels are not


def test_ips_expected_saves_matches_truth_naive_does_not():
    rng = np.random.default_rng(2)

    def p(k):
        return debias.power_law(k, 2.0, 0.8)

    sessions = []
    for _ in range(3000):
        true_save = rng.random(25) * 0.4
        logged_pos = np.arange(25)
        save = (rng.random(25) < p(logged_pos) * true_save).astype(float)
        sessions.append({"save": save, "pos": logged_pos, "true_save": true_save})
    scores = [s["true_save"] for s in sessions]  # evaluate the ideal ranking
    m = debias.counterfactual_metrics(sessions, scores, p, p)
    assert abs(m["ips_expected_saves"] / m["true_expected_saves"] - 1) < 0.03
    assert m["naive_expected_saves"] < 0.7 * m["true_expected_saves"]
