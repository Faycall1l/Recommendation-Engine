"""Trivial baselines that give every SOTA eval honest lower bounds.

Mean baselines (Global/User/Item) predict explicit ratings; MostPopular and
Random produce rankings. None of them should beat a real collaborative filter
on its home turf — if one does, the eval protocol is broken.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class _MeanBaseline:
    """Shared mechanics for per-entity mean predictors."""

    axis: int | None = None  # 0 -> items, 1 -> users

    def __init__(self):
        self.matrix: sp.csr_matrix | None = None
        self.means: np.ndarray | None = None
        self.global_mean: float = 0.0

    def fit(self, matrix: sp.csr_matrix) -> _MeanBaseline:
        self.matrix = matrix.tocsr()
        if self.axis == 1:  # user means
            counts = self.matrix.getnnz(axis=1)
            sums = np.asarray(self.matrix.sum(axis=1)).ravel()
        else:  # item means
            counts = self.matrix.getnnz(axis=0)
            sums = np.asarray(self.matrix.sum(axis=0)).ravel()
        self.means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
        self.global_mean = float(sums.sum() / max(self.matrix.nnz, 1))
        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        raise NotImplementedError


class GlobalMean(_MeanBaseline):
    """Predict every rating with the dataset-wide mean."""

    def fit(self, matrix: sp.csr_matrix) -> GlobalMean:
        self.matrix = matrix.tocsr()
        self.global_mean = float(self.matrix.sum() / max(self.matrix.nnz, 1))
        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        return float(self.global_mean)

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10):
        raise NotImplementedError("GlobalMean is a rating-prediction baseline only")


class UserMean(_MeanBaseline):
    """Predict each rating with that user's own mean."""

    axis = 1

    def fit(self, matrix: sp.csr_matrix) -> UserMean:
        super().fit(matrix)
        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        return float(self.means[user_idx])

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10):
        raise NotImplementedError("UserMean is a rating-prediction baseline only")


class ItemMean(_MeanBaseline):
    """Predict each rating with that item's mean."""

    axis = 0

    def fit(self, matrix: sp.csr_matrix) -> ItemMean:
        super().fit(matrix)
        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        return float(self.means[item_idx])

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10):
        raise NotImplementedError("ItemMean is a rating-prediction baseline only")


class MostPopular:
    """Rank by how many users rated each item — the classic cold-start prior."""

    def __init__(self):
        self.counts: np.ndarray | None = None
        self.item_order: np.ndarray | None = None

    def fit(self, matrix: sp.csr_matrix) -> MostPopular:
        matrix = matrix.tocsr()
        self.counts = matrix.getnnz(axis=0).astype(float)
        self.item_order = np.argsort(-self.counts)
        return self

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        rated = set(matrix.indices[matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]])
        out: list[tuple[int, float]] = []
        for item_idx in self.item_order:
            if len(out) == n:
                break
            if item_idx in rated:
                continue
            out.append((int(item_idx), float(self.counts[item_idx])))
        return out

    def predict(self, user_idx: int, item_idx: int) -> float:
        raise NotImplementedError("MostPopular is a ranking baseline only")


class RandomBaseline:
    """Random unseen items — the theoretical floor for any ranking."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng: np.random.Generator | None = None

    def fit(self, matrix: sp.csr_matrix) -> RandomBaseline:
        self.rng = np.random.default_rng(self.seed)
        return self

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        rated = set(matrix.indices[matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]])
        candidates = np.fromiter(
            (idx for idx in range(matrix.shape[1]) if idx not in rated),
            dtype=int,
            count=matrix.shape[1] - len(rated),
        )
        picks = self.rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        return [(int(idx), 0.0) for idx in picks]

    def predict(self, user_idx: int, item_idx: int) -> float:
        raise NotImplementedError("RandomBaseline is a ranking baseline only")
