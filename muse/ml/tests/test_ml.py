import numpy as np
from muse_ml import embeddings, features, pinnersage, ranker, spam


def test_auc_matches_definition():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    pairs = [(i, j) for i in range(6) for j in range(6) if y[i] == 1 and y[j] == 0]
    brute = np.mean([s[i] > s[j] for i, j in pairs])
    assert abs(ranker.auc(y, s) - brute) < 1e-12


def test_mlp_gradients_learn_separable_task():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 5))
    Y = np.column_stack([(X[:, 0] > 0), (X[:, 1] + X[:, 2] > 0), (X[:, 3] < -1)]).astype(float)
    m = ranker.MultiTaskMLP(5, hidden=16).fit(X, Y, epochs=15, lr=1e-2)
    p = m.predict(X)
    for h in range(3):
        assert ranker.auc(Y[:, h], p[:, h]) > 0.97


def test_ndcg_perfect_and_reversed():
    rel = np.array([1.0, 0, 0, 1, 0])
    assert ranker.ndcg_at_k([(rel, rel)], k=5) == 1.0
    assert ranker.ndcg_at_k([(rel, -rel)], k=5) < 0.6


def test_minhash_lsh_groups_near_duplicates_only():
    titles = [
        "Boho reading nook ideas",
        "Boho reading nook ideas shop now",
        "Boho reading nook ideas",
        "Vegan one-pot curry dinner",
        "Brutalist cabin facade",
    ]
    cl = spam.near_duplicate_clusters(titles)
    assert cl[0] == cl[2]
    assert len({cl[0], cl[3], cl[4]}) == 3


def test_cold_pins_get_content_projected_embeddings():
    rng = np.random.default_rng(0)
    vis = rng.normal(size=(300, 8))
    vis /= np.linalg.norm(vis, axis=1, keepdims=True)
    graph = vis @ rng.normal(size=(8, 6))
    counts = np.r_[np.full(250, 10.0), np.zeros(50)]
    W = embeddings.fit_content_projection(vis, graph, counts)
    E = embeddings.unified_embeddings(vis, np.where(counts[:, None] > 0, graph, 0), counts, W)
    assert np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-5)
    g_true = graph[250:] / np.linalg.norm(graph[250:], axis=1, keepdims=True)
    g_cold = E[250:, :6] / np.linalg.norm(E[250:, :6], axis=1, keepdims=True)
    assert np.mean(np.sum(g_true * g_cold, 1)) > 0.95  # projection recovers graph space


def test_pinnersage_separates_two_interests():
    rng = np.random.default_rng(1)
    a, b = np.eye(16)[0], np.eye(16)[1]
    E = np.vstack([a + 0.05 * rng.normal(size=(20, 16)), b + 0.05 * rng.normal(size=(20, 16))])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    cl = pinnersage.user_clusters(np.arange(40), np.full(40, 80), E, ref_day=86)
    assert len(cl) == 2


def test_featurize_empty_user_is_safe():
    emb = np.eye(4, 6)
    vis = np.eye(4, 3)
    topics = np.array([0, 1, 2, 3])
    us = features.UserState([], [], [], emb, vis, topics)
    static = features.pin_static(np.array([0, 1, 2, 3.0]), np.array([80, 10, 0, 86]), np.zeros(4))
    X = features.featurize(us, [0, 3], emb, vis, topics, static)
    assert X.shape == (2, len(features.FEATURES)) and np.all(X[:, :4] == 0)
