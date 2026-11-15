"""Pin embeddings: collaborative (board co-save graph) + content (visual), unified.

* graph embedding  - truncated SVD of the PPMI-weighted pin x board matrix
                     (pins saved to the same boards end up close together).
* cold start       - pins with few/no saves have no graph signal, so we learn a
                     ridge projection visual -> graph space on warm pins and
                     blend it in by a confidence weight n / (n + k).
* unified          - [graph ; beta * visual], L2-normalised. This is what the
                     ANN index, PinnerSage clusters and the ranker all use.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

DIM_GRAPH = 32


def _unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def graph_embeddings(saves: np.ndarray, n_pins: int, n_boards: int, dim: int = DIM_GRAPH, seed: int = 0) -> np.ndarray:
    rows, cols = saves[:, 2], saves[:, 1]
    X = csr_matrix((np.ones(len(saves)), (rows, cols)), shape=(n_pins, n_boards))
    X.data[:] = 1.0  # duplicate saves count once
    total = X.sum()
    pin_tot = np.asarray(X.sum(1)).ravel()
    board_tot = np.asarray(X.sum(0)).ravel()
    X = X.tocoo()
    pmi = np.log(X.data * total / (pin_tot[X.row] * board_tot[X.col] ** 0.75 * total**0.25 + 1e-9))
    keep = pmi > 0
    P = csr_matrix((pmi[keep], (X.row[keep], X.col[keep])), shape=(n_pins, n_boards))
    u, s, _ = svds(P, k=dim, random_state=seed)
    return (u * np.sqrt(s)).astype(np.float32)


def fit_content_projection(
    visual: np.ndarray, graph: np.ndarray, counts: np.ndarray, min_saves: int = 3, lam: float = 1.0
) -> np.ndarray:
    warm = counts >= min_saves
    Xv, Yg = visual[warm], _unit(graph[warm])
    A = Xv.T @ Xv + lam * np.eye(Xv.shape[1])
    return np.linalg.solve(A, Xv.T @ Yg).astype(np.float32)  # (Dv, Dg)


def unified_embeddings(
    visual: np.ndarray,
    graph: np.ndarray,
    counts: np.ndarray,
    W: np.ndarray | None,
    k_conf: float = 3.0,
    beta: float = 0.5,
) -> np.ndarray:
    g = _unit(graph)
    if W is not None:
        g_proj = _unit(visual @ W)
        conf = (counts / (counts + k_conf))[:, None]
        g = _unit(conf * g + (1 - conf) * g_proj)
    else:
        g = np.where((counts > 0)[:, None], g, 0.0)
    return _unit(np.concatenate([g, beta * visual], axis=1)).astype(np.float32)
