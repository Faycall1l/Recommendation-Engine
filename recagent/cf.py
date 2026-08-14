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

from typing import Any

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
        self.similarity_norm = self._normalized_similarity()
        return self

    def _normalized_similarity(self) -> np.ndarray:
        """Rows of the similarity matrix divided by their L1 (|sim|) weight."""
        row_sums = np.abs(self.similarity).sum(axis=1)
        normalized = np.zeros_like(self.similarity)
        np.divide(self.similarity, row_sums[:, None], out=normalized, where=row_sums[:, None] != 0)
        return normalized

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

    def score_all(self) -> np.ndarray:
        """Dense (n_users x n_items) predicted ratings for every user.

        Matches per-user ``predict``/``recommend`` scoring in one batched op.
        """
        return self.user_means[:, None] + self.similarity_norm @ self.centered

    def save(self, path: str | Any) -> UserBasedCF:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        coo = self.centered.tocoo()
        np.savez(
            path,
            min_sim=self.min_sim,
            user_means=self.user_means,
            similarity=self.similarity,
            c_data=coo.data,
            c_row=coo.row,
            c_col=coo.col,
            c_shape=np.asarray(coo.shape),
        )
        return self

    @classmethod
    def load(cls, path: str | Any) -> UserBasedCF:
        saved = np.load(path)
        obj = cls(min_sim=float(saved["min_sim"]))
        obj.user_means = saved["user_means"]
        obj.centered = sp.coo_matrix(
            (saved["c_data"], (saved["c_row"], saved["c_col"])),
            shape=tuple(saved["c_shape"]),
        ).tocsr()
        obj.similarity = saved["similarity"]
        obj.similarity_norm = obj._normalized_similarity()
        return obj


class ItemBasedCF:
    """Item-item neighbourhood collaborative filtering with adjusted cosine."""

    def __init__(self, min_sim: float = 0.0):
        self.min_sim = min_sim
        self.matrix: sp.csr_matrix | None = None
        self.user_means: np.ndarray | None = None
        self.item_means: np.ndarray | None = None
        self.centered: sp.csr_matrix | None = None
        self.similarity: np.ndarray | None = None

    def fit(self, matrix: sp.csr_matrix) -> ItemBasedCF:
        self.matrix = matrix.tocsr()
        counts = self.matrix.getnnz(axis=0)
        sums = np.asarray(self.matrix.sum(axis=0)).ravel()
        item_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
        centered = self.matrix.copy()
        nonzero_rows, nonzero_cols = centered.nonzero()
        centered[nonzero_rows, nonzero_cols] -= item_means[nonzero_cols]
        self.item_means = item_means
        self.centered = centered
        user_counts = np.maximum(self.matrix.getnnz(axis=1), 1)
        self.user_means = np.asarray(self.matrix.sum(axis=1)).ravel() / user_counts
        self.similarity = self._similarity()
        return self

    def _similarity(self) -> np.ndarray:
        """Adjusted cosine: L2-normalize mean-centered columns, then C^T C."""
        squared = self.centered.multiply(self.centered)
        norms = np.sqrt(np.asarray(squared.sum(axis=0)).ravel())
        inv = np.zeros_like(norms)
        np.divide(1.0, norms, out=inv, where=norms > 0)
        normalized = self.centered @ sp.diags(inv)  # column-normalized
        similarity = (normalized.T @ normalized).toarray()
        np.fill_diagonal(similarity, 0.0)
        similarity[similarity < self.min_sim] = 0.0
        return similarity

    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predicted rating: similarity-weighted mean of the user's own ratings."""
        if self.matrix is None or self.similarity is None:
            raise ValueError("fit() must be called before predict()")
        similarity = self.similarity[item_idx]
        row_start, row_end = self.matrix.indptr[user_idx : user_idx + 2]
        rated_items = self.matrix.indices[row_start:row_end]
        rated_values = self.matrix.data[row_start:row_end]
        weights = similarity[rated_items]
        numer = float(np.dot(weights, rated_values))
        denom = float(np.abs(weights).sum())
        if denom == 0.0:
            return float(self.user_means[user_idx])
        return numer / denom
