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
