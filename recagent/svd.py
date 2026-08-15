"""Biased matrix factorization (SVD-class) from scratch.

Adds the global mean plus per-user/per-item bias terms to the ALS objective,
the classic improvement over unit-weight ALS (Funk/Koren-style biased MF).
The RMSE 0.89-0.95 figures published for MovieLens are achieved by exactly
this class of model, so this is the engine the ml-100k "published range" line
in the findings refers to — and the one that should close the gap vs the
memory-based engines on rating prediction.

Alternates per-entity closed-form solves for the factor matrices against the
residuals, then re-derives each entity's bias from those residuals with
shrinkage (the per-user/per-item offsets are updated as a by-product of each
entity's solve, exactly as in the standard biased-ALS scheme).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp


class BiasedMF:
    def __init__(
        self,
        factors: int = 32,
        iterations: int = 20,
        regularization: float = 0.1,
        bias_shrinkage: float = 25.0,
        seed: int = 42,
    ):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.bias_shrinkage = bias_shrinkage
        self.seed = seed
        self.mu = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, matrix: sp.csr_matrix) -> BiasedMF:
        matrix = matrix.tocsr()
        n_users, n_items = matrix.shape
        rng = np.random.default_rng(self.seed)
        self.mu = float(matrix.data.mean())
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.user_factors = rng.normal(0.0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0.0, 0.01, (n_items, self.factors))
        lam = self.regularization
        lam_b = self.bias_shrinkage
        for _ in range(self.iterations):
            self.item_factors, self.item_bias = self._solve(
                matrix.T.tocsr(),
                self.user_factors,
                self.user_bias,
                n_items,
                lam,
                lam_b,
            )
            self.user_factors, self.user_bias = self._solve(
                matrix,
                self.item_factors,
                self.item_bias,
                n_users,
                lam,
                lam_b,
            )
        return self

    def _solve(
        self,
        view: sp.csr_matrix,
        latent: np.ndarray,
        other_bias: np.ndarray,
        n_entities: int,
        lam: float,
        lam_b: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One factor+bias pass for the entities of ``view`` (rows)."""
        eye = lam * np.eye(self.factors)
        updated = np.zeros((n_entities, self.factors))
        bias = np.zeros(n_entities)
        for entity in range(n_entities):
            start, end = view.indptr[entity], view.indptr[entity + 1]
            cols = view.indices[start:end]
            values = view.data[start:end]
            if end == start:
                continue
            residuals = values - (self.mu + other_bias[cols])
            rated_latent = latent[cols]
            a = rated_latent.T @ rated_latent + eye
            factors = np.linalg.solve(a, rated_latent.T @ residuals)
            updated[entity] = factors
            bias[entity] = (residuals - rated_latent @ factors).sum() / (len(cols) + lam_b)
        return updated, bias

    def predict(self, user_idx: int, item_idx: int) -> float:
        return float(
            self.mu
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
            + self.user_factors[user_idx] @ self.item_factors[item_idx]
        )

    def _scores(self, user_idx: int) -> np.ndarray:
        return (
            self.mu
            + self.user_bias[user_idx]
            + self.item_bias
            + self.user_factors[user_idx] @ self.item_factors.T
        )

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        scores = self._scores(user_idx)
        rated = set(matrix.indices[matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]])
        out: list[tuple[int, float]] = []
        for item_idx in np.argsort(-scores):
            if len(out) == n:
                break
            if item_idx in rated:
                continue
            out.append((int(item_idx), float(scores[item_idx])))
        return out

    def save(self, path: str | Any) -> BiasedMF:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            bias_shrinkage=self.bias_shrinkage,
            seed=self.seed,
            mu=self.mu,
            user_bias=self.user_bias,
            item_bias=self.item_bias,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
        )
        return self

    @classmethod
    def load(cls, path: str | Any) -> BiasedMF:
        saved = np.load(path)
        obj = cls(
            factors=int(saved["factors"]),
            iterations=int(saved["iterations"]),
            regularization=float(saved["regularization"]),
            bias_shrinkage=float(saved["bias_shrinkage"]),
            seed=int(saved["seed"]),
        )
        obj.mu = float(saved["mu"])
        obj.user_bias = saved["user_bias"]
        obj.item_bias = saved["item_bias"]
        obj.user_factors = saved["user_factors"]
        obj.item_factors = saved["item_factors"]
        return obj
