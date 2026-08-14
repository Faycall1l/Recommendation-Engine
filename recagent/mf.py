"""Explicit matrix factorization via alternating least squares, from scratch.

Minimises sum over observed ratings of ``(r_ui - u_u^T v_i)^2`` plus an L2
penalty, alternating closed-form solves for the user and item factor matrices.
This is the classic SOTA-class predictor for explicit-rating datasets
(ml-100k RMSE ~0.89-0.95), implemented here with no external recommender
library.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


class ExplicitALS:
    def __init__(self, factors: int = 64, iterations: int = 20, regularization: float = 0.1, seed: int = 42):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.seed = seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, matrix: sp.csr_matrix) -> ExplicitALS:
        matrix = matrix.tocsr()
        n_users, n_items = matrix.shape
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0.0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0.0, 0.01, (n_items, self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._solve(matrix, self.item_factors, n_users)
            self.item_factors = self._solve(matrix.T.tocsr(), self.user_factors, n_items)
        return self

    def _solve(self, view: sp.csr_matrix, latent: np.ndarray, n_entities: int) -> np.ndarray:
        """Closed-form least squares for one side given the other.

        ``view`` must expose one row per entity (users for the user solve, the
        transposed matrix for the item solve).
        """
        updated = np.zeros((n_entities, self.factors))
        eye = self.regularization * np.eye(self.factors)
        for entity in range(n_entities):
            start, end = view.indptr[entity], view.indptr[entity + 1]
            cols = view.indices[start:end]
            values = view.data[start:end]
            if end == start:
                continue
            latent_rated = latent[cols]
            a = latent_rated.T @ latent_rated + eye
            updated[entity] = np.linalg.solve(a, latent_rated.T @ values)
        return updated

    def predict(self, user_idx: int, item_idx: int) -> float:
        return float(self.user_factors[user_idx] @ self.item_factors[item_idx])

    def score_all(self) -> np.ndarray:
        return self.user_factors @ self.item_factors.T

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        scores = self.user_factors[user_idx] @ self.item_factors.T
        rated = set(matrix.indices[matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]])
        out: list[tuple[int, float]] = []
        for item_idx in np.argsort(-scores):
            if len(out) == n:
                break
            if item_idx in rated:
                continue
            out.append((int(item_idx), float(scores[item_idx])))
        return out

    def save(self, path: str | Any) -> ExplicitALS:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            seed=self.seed,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
        )
        return self

    @classmethod
    def load(cls, path: str | Any) -> ExplicitALS:
        saved = np.load(path)
        obj = cls(
            factors=int(saved["factors"]),
            iterations=int(saved["iterations"]),
            regularization=float(saved["regularization"]),
            seed=int(saved["seed"]),
        )
        obj.user_factors = saved["user_factors"]
        obj.item_factors = saved["item_factors"]
        return obj
