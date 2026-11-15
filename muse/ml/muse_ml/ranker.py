"""Multi-task ranker: one shared MLP trunk, three sigmoid heads (click, save, hide).

Pure NumPy (forward + backprop + Adam) so the exact same arithmetic can be
re-implemented in Go for serving without an ML runtime dependency.
"""

from __future__ import annotations

import json

import numpy as np

HEADS = ["click", "save", "hide"]


class MultiTaskMLP:
    def __init__(self, n_in: int, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(1 / hidden), (hidden, len(HEADS)))
        self.b2 = np.zeros(len(HEADS))
        self.mu = np.zeros(n_in)
        self.sd = np.ones(n_in)

    def _forward(self, Xn):
        h = np.maximum(Xn @ self.W1 + self.b1, 0)
        return h, 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))

    def predict(self, X):
        return self._forward((X - self.mu) / self.sd)[1]

    def fit(self, X, Y, epochs=8, lr=3e-3, batch=512, head_weights=(1.0, 2.0, 1.0), seed=0):
        rng = np.random.default_rng(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        hw = np.asarray(head_weights)
        t = 0
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s : s + batch]
                h, p = self._forward(Xn[b])
                g_logit = (p - Y[b]) * hw / len(b)  # dBCE/dlogit
                gW2 = h.T @ g_logit
                gb2 = g_logit.sum(0)
                gh = (g_logit @ self.W2.T) * (h > 0)
                gW1 = Xn[b].T @ gh
                gb1 = gh.sum(0)
                t += 1
                for i, g in enumerate([gW1, gb1, gW2, gb2]):
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    params[i] -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
        return self

    def to_json(self, features: list[str]) -> str:
        return json.dumps(
            {
                "features": features,
                "heads": HEADS,
                "mu": self.mu.tolist(),
                "sd": self.sd.tolist(),
                "W1": self.W1.tolist(),
                "b1": self.b1.tolist(),
                "W2": self.W2.tolist(),
                "b2": self.b2.tolist(),
            }
        )


def auc(y, s):
    order = np.argsort(s)
    y = np.asarray(y)[order]
    n1 = y.sum()
    n0 = len(y) - n1
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def ndcg_at_k(groups, k=10):
    """groups: list of (relevance array, score array)."""
    vals = []
    for rel, sc in groups:
        if rel.sum() == 0:
            continue
        disc = 1 / np.log2(np.arange(2, k + 2))
        top = rel[np.argsort(-sc)][:k]
        ideal = np.sort(rel)[::-1][:k]
        vals.append((top * disc[: len(top)]).sum() / (ideal * disc[: len(ideal)]).sum())
    return float(np.mean(vals))
