"""Collaborative filtering engine backed by weighted alternating least squares."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from recagent.data import encode, fetch_movielens, leave_one_out, load_items, load_ratings


class Recommender:
    """A thin, dependency-light wrapper around implicit ALS."""

    def __init__(self, factors: int = 64, iterations: int = 20, regularization: float = 0.1):
        self.model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            use_gpu=False,
            random_state=42,
        )

    def fit(self, matrix: sp.csr_matrix) -> Recommender:
        self.model.fit(matrix, show_progress=False)
        return self

    def recommend(
        self, matrix: sp.csr_matrix, user_idx: int, n: int = 10
    ) -> list[tuple[int, float]]:
        """Top-n items for a user, filtering out already-interacted items."""
        ids, scores = self.model.recommend(
            user_idx, matrix[user_idx], N=n, filter_already_liked_items=True
        )
        return list(zip(map(int, ids), map(float, scores)))

    def similar_items(self, item_idx: int, n: int = 10) -> list[tuple[int, float]]:
        ids, scores = self.model.similar_items(item_idx, N=n)
        return list(zip(map(int, ids), map(float, scores)))

    def similar_users(self, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        ids, scores = self.model.similar_users(user_idx, N=n)
        return list(zip(map(int, ids), map(float, scores)))

    def save(self, path: str | Path) -> Recommender:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        return self

    @classmethod
    def load(cls, path: str | Path) -> Recommender:
        # implicit's public name is a factory, so go through a concrete
        # instance to reach the classmethod that restores factor matrices.
        factory = AlternatingLeastSquares(use_gpu=False)
        restored = factory.__class__.load(path)
        return cls._from_model(restored)

    @classmethod
    def _from_model(cls, model: AlternatingLeastSquares) -> Recommender:
        obj = cls.__new__(cls)
        obj.model = model
        return obj


def train_from_data(
    data_dir: str | Path = "data",
    *,
    min_interactions: int = 5,
    seed: int = 42,
    factors: int = 64,
    iterations: int = 20,
    cf: str = "user",
) -> tuple[Recommender, sp.csr_matrix, dict, dict, dict, dict, np.ndarray, np.ndarray]:
    """End-to-end: fetch, split leave-one-out, encode, fit a CF engine.

    ``cf`` selects the engine: ``als`` (implicit ALS wrapper) or the
    memory-based ``user``/``item`` neighbourhood methods (default ``user``).

    Returns the fitted model plus the artefacts needed for inference and eval.
    """
    from recagent.cf import CF_KINDS, build_cf

    if cf not in CF_KINDS:
        raise ValueError(f"cf must be one of {CF_KINDS}, got {cf!r}")
    dataset_dir = fetch_movielens(data_dir)
    users, items, ratings = load_ratings(dataset_dir)
    items_meta = load_items(dataset_dir)
    (tr_u, tr_i, tr_r), _ = leave_one_out(
        users, items, ratings, min_interactions=min_interactions, seed=seed
    )
    matrix, uid_to_idx, iid_to_idx, user_ids, item_ids = encode(tr_u, tr_i, tr_r)
    if cf == "als":
        recommender: Recommender = Recommender(factors=factors, iterations=iterations).fit(matrix)
    else:
        recommender = build_cf(cf, matrix)
    return (
        recommender,
        matrix,
        uid_to_idx,
        iid_to_idx,
        user_ids,
        item_ids,
        items_meta,
    )
