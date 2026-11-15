"""Spam / low-quality pin detection.

1. MinHash + LSH banding over title character shingles finds near-duplicate
   clusters in ~linear time (spam rings mass-post templated titles).
2. A logistic-regression classifier over cluster/creator/engagement features
   is trained on a small "human-reviewed" label sample (15% of pins) and
   scores every pin. The score feeds the ranker and a hard serving filter.
"""

from __future__ import annotations

import re
import zlib
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

LEXICON = {"free", "click", "$$$", "deal", "offer", "won't", "100%", "gift", "link", "believe"}
_P = (1 << 61) - 1


def shingles(text: str, k: int = 4) -> set[int]:
    t = re.sub(r"\s+", " ", text.lower())
    return {zlib.crc32(t[i : i + k].encode()) for i in range(max(1, len(t) - k + 1))}


def minhash(sh: set[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.fromiter(sh, dtype=np.uint64)
    return ((np.outer(a, x) + b[:, None]) % _P).min(1)


def near_duplicate_clusters(titles: list[str], n_hash: int = 64, bands: int = 16, seed: int = 0) -> np.ndarray:
    """Returns a cluster id per title (union-find over LSH bucket collisions)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 1 << 31, size=n_hash, dtype=np.uint64)
    b = rng.integers(0, 1 << 31, size=n_hash, dtype=np.uint64)
    sigs = np.array([minhash(shingles(t), a, b) for t in titles])
    parent = list(range(len(titles)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    r = n_hash // bands
    for band in range(bands):
        buckets = defaultdict(list)
        for i, s in enumerate(sigs[:, band * r : (band + 1) * r]):
            buckets[s.tobytes()].append(i)
        for members in buckets.values():
            for j in members[1:]:
                parent[find(j)] = find(members[0])
    return np.array([find(i) for i in range(len(titles))])


def spam_features(
    pins: list[dict], impressions: np.ndarray, exam_weights: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    titles = [p["title"] for p in pins]
    cl = near_duplicate_clusters(titles)
    cl_size = np.bincount(cl)[cl]
    creators = np.array([p["creator"] for p in pins])
    _, cinv, ccount = np.unique(creators, return_inverse=True, return_counts=True)
    dup_by_creator = np.bincount(cinv, weights=(cl_size > 20)) / ccount
    lex = np.array([sum(w in LEXICON for w in t.lower().split()) for t in titles])
    w = np.ones(len(impressions)) if exam_weights is None else exam_weights
    n = np.bincount(impressions[:, 1], weights=w, minlength=len(pins))  # expected examinations
    hide = np.bincount(impressions[:, 1], weights=impressions[:, 4], minlength=len(pins))
    save = np.bincount(impressions[:, 1], weights=impressions[:, 3], minlength=len(pins))
    X = np.column_stack(
        [
            np.log1p(cl_size),
            dup_by_creator[cinv],
            lex,
            (hide + 0.2) / (n + 10),
            (save + 1.5) / (n + 10),
            np.log1p(ccount[cinv]),
        ]
    )
    return X, ["log_dup_cluster", "creator_dup_ratio", "lexicon_hits", "hide_rate", "save_rate", "log_creator_pins"]


def train_spam_model(X: np.ndarray, y: np.ndarray, label_frac: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    labeled = rng.random(len(y)) < label_frac
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(X[labeled], y[labeled])
    return clf, labeled
