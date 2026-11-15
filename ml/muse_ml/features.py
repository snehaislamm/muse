"""Ranking features. This file is the *spec*: serving/internal/rank/features.go must
compute exactly the same numbers (enforced by the parity fixture test in Go).

User state is derived only from the user's saves (pin ids + days), so the Go
service can rebuild it from the same artifacts and update it online.
"""

from __future__ import annotations

import numpy as np

FEATURES = ["sim_mean", "sim_cluster_max", "visual_sim", "topic_affinity", "log_popularity", "age", "spam_score"]
REF_DAY = 86
HALF_LIFE = 14.0
N_TOPICS = 16
D_GRAPH = 32  # first D_GRAPH dims of the unified embedding are the graph part


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class UserState:
    """mean (decayed) unified vector, cluster centroids, mean visual vector, topic mix."""

    def __init__(self, pin_ids, days, clusters, emb, visual, topic_ids):
        pin_ids = np.asarray(pin_ids, dtype=np.int64)
        days = np.asarray(days, dtype=np.float64)
        self.empty = len(pin_ids) == 0
        if self.empty:
            self.mean = np.zeros(emb.shape[1])
            self.vis = np.zeros(visual.shape[1])
            self.centroids = np.zeros((0, emb.shape[1]))
            self.topic = np.zeros(N_TOPICS)
            return
        w = np.exp(-np.log(2) * (REF_DAY - days) / HALF_LIFE)
        self.mean = _unit((emb[pin_ids].astype(np.float64) * w[:, None]).sum(0))
        self.vis = _unit((visual[pin_ids].astype(np.float64) * w[:, None]).sum(0))
        cents = [_unit(emb[np.asarray(c)].astype(np.float64).mean(0)) for c in clusters if len(c)]
        self.centroids = np.array(cents) if cents else self.mean[None, :]
        t = np.bincount(topic_ids[pin_ids], weights=w, minlength=N_TOPICS)
        self.topic = t / t.sum()


def pin_static(save_counts, created_day, spam_score):
    return {
        "log_popularity": np.log1p(save_counts),
        "age": np.clip((REF_DAY - created_day) / 90.0, 0.0, 1.0),
        "spam_score": spam_score,
    }


def featurize(us: UserState, pin_ids, emb, visual, topic_ids, static) -> np.ndarray:
    pin_ids = np.asarray(pin_ids, dtype=np.int64)
    E = emb[pin_ids].astype(np.float64)
    out = np.zeros((len(pin_ids), len(FEATURES)))
    if not us.empty:
        out[:, 0] = E @ us.mean
        out[:, 1] = (E @ us.centroids.T).max(1)
        out[:, 2] = visual[pin_ids].astype(np.float64) @ us.vis
        out[:, 3] = us.topic[topic_ids[pin_ids]]
    out[:, 4] = static["log_popularity"][pin_ids]
    out[:, 5] = static["age"][pin_ids]
    out[:, 6] = static["spam_score"][pin_ids]
    return out
