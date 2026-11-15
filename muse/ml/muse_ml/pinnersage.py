"""PinnerSage-style multi-interest user representation.

A single averaged user vector lands *between* a user's interests (the mean of
"sourdough" and "brutalist cabins" matches neither). Instead we cluster each
user's saves with Ward linkage, represent every cluster by its medoid pin
(an actual pin, so it is always on the embedding manifold) and weight clusters
by exponentially time-decayed activity.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def user_clusters(
    pin_ids: np.ndarray,
    days: np.ndarray,
    emb: np.ndarray,
    ref_day: int,
    max_clusters: int = 5,
    dist_threshold: float = 1.1,
    half_life: float = 14.0,
) -> list[dict]:
    if len(pin_ids) == 0:
        return []
    decay = np.exp(-np.log(2) * (ref_day - days) / half_life)
    if len(pin_ids) == 1:
        labels = np.array([1])
    else:
        Z = linkage(emb[pin_ids], method="ward")
        labels = fcluster(Z, t=dist_threshold, criterion="distance")
    out = []
    for lab in np.unique(labels):
        m = labels == lab
        members = pin_ids[m]
        E = emb[members]
        d = 1 - E @ E.T
        medoid = int(members[np.argmin(d.sum(1))])
        out.append({"medoid": medoid, "importance": float(decay[m].sum()), "size": int(m.sum())})
    out.sort(key=lambda c: -c["importance"])
    return out[:max_clusters]


def mean_vector(pin_ids, days, emb, ref_day, half_life=14.0):
    w = np.exp(-np.log(2) * (ref_day - days) / half_life)
    v = (emb[pin_ids] * w[:, None]).sum(0)
    return v / max(np.linalg.norm(v), 1e-9)
