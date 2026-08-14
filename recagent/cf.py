"""Memory-based collaborative filtering, implemented from scratch in numpy.

Classic neighbourhood methods over the explicit-rating matrix:

- ``UserBasedCF`` — Pearson-correlated nearest users; predict a user's rating
  of an item as their mean plus the similarity-weighted average of neighbour
  deviations.
- ``ItemBasedCF`` — adjusted-cosine item similarity; predict from the user's
  own ratings of similar items.

Both expose the same interface as the ALS ``Recommender``
(``fit`` / ``recommend(matrix, user_idx, n)``) so the rest of the pipeline —
tool registry, CLI, eval, agent — treats them identically.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

CF_KINDS = ("als", "user", "item")


class UserBasedCF:
    """User-user neighbourhood collaborative filtering with Pearson correlation."""

    def __init__(self, min_sim: float = 0.0):
        self.min_sim = min_sim
        self.matrix: sp.csr_matrix | None = None
        self.user_means: np.ndarray | None = None
        self.centered: sp.csr_matrix | None = None
        self.similarity: np.ndarray | None = None
        self.similarity_norm: np.ndarray | None = None

    def fit(self, matrix: sp.csr_matrix) -> UserBasedCF:
        self.matrix = matrix.tocsr()
        counts = self.matrix.getnnz(axis=1)
        sums = np.asarray(self.matrix.sum(axis=1)).ravel()
        user_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
        # Mean-center each user's ratings; zeros stay zeros (a user with no
        # interaction gets a mean of 0 but we never score them).
        centered = self.matrix.copy()
        nonzero_rows, nonzero_cols = centered.nonzero()
        centered[nonzero_rows, nonzero_cols] -= user_means[nonzero_rows]
        self.user_means = user_means
        self.centered = centered
        self.similarity = self._similarity()
        return self

    def _similarity(self) -> np.ndarray:
        """Dense user-user Pearson similarity via cosine on centered ratings."""
        squared = self.centered.multiply(self.centered)
        norms = np.sqrt(np.asarray(squared.sum(axis=1)).ravel())
        inv = np.zeros_like(norms)
        np.divide(1.0, norms, out=inv, where=norms > 0)
        normalized = sp.diags(inv) @ self.centered  # L2-normalized rows
        similarity = (normalized @ normalized.T).toarray()
        np.fill_diagonal(similarity, 0.0)
        similarity[similarity < self.min_sim] = 0.0
        return similarity

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predicted rating: user mean + similarity-weighted neighbour deviation."""
        deviations = self.centered[:, item_idx].toarray().ravel()
        similarity = self.similarity[user_idx]
        denom = float(np.abs(similarity).sum())
        if denom == 0.0:
            return float(self.user_means[user_idx])
        return float(self.user_means[user_idx] + (similarity @ deviations) / denom)

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        """Top-n unseen items by predicted rating, best first (ALS-compatible)."""
        similarity = self.similarity[user_idx]
        denom = float(np.abs(similarity).sum())
        if denom == 0.0:
            scores = np.full(matrix.shape[1], self.user_means[user_idx])
        else:
            scores = self.user_means[user_idx] + (similarity / denom) @ self.centered
        rated = set(matrix.indices[matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]])
        out: list[tuple[int, float]] = []
        for item_idx in np.argsort(-scores):
            if len(out) == n:
                break
            if item_idx in rated:
                continue
            out.append((int(item_idx), float(scores[item_idx])))
        return out
