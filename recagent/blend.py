"""Reciprocal-rank fusion of a personalized engine with global popularity.

Plain LOO rewards recommending popular items (see FINDINGS §2): a *personal*
model like ALS can be lifted by fusing its ranking with the popularity prior.
RRF (Cormack et al., SIGIR 2009) merges two rank lists without needing either
system's raw scores — only ``recommend()`` — so it composes with any engine.
The ``base_weight`` controls how much the personalized list dominates.
"""

from __future__ import annotations

from typing import Any

import scipy.sparse as sp

from recagent.baselines import MostPopular


class RankBlend:
    def __init__(
        self,
        base_kind: str = "als",
        base_weight: float = 0.6,
        top_k: int = 200,
        k: int = 60,
        seed: int = 42,
        factors: int = 64,
        iterations: int = 20,
        regularization: float = 0.1,
    ):
        self.base_kind = base_kind
        self.base_weight = base_weight
        self.top_k = top_k
        self.k = k
        self.seed = seed
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.base: Any | None = None
        self.popularity: MostPopular | None = None

    def fit(self, matrix: sp.csr_matrix) -> RankBlend:
        from recagent.engines import build_engine

        self.base = build_engine(
            self.base_kind,
            matrix,
            seed=self.seed,
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
        )
        self.popularity = MostPopular().fit(matrix)
        return self

    def recommend(self, matrix: sp.csr_matrix, user_idx: int, n: int = 10) -> list[tuple[int, float]]:
        base_ranked = {
            int(item): rank
            for rank, (item, _score) in enumerate(self.base.recommend(matrix, user_idx, n=self.top_k), 1)
        }
        pop_ranked = {
            int(item): rank
            for rank, (item, _score) in enumerate(self.popularity.recommend(matrix, user_idx, n=self.top_k), 1)
        }
        base_beyond = len(base_ranked) + 1
        pop_beyond = len(pop_ranked) + 1
        scored: dict[int, float] = {}
        for item in set(base_ranked) | set(pop_ranked):
            scored[item] = (
                self.base_weight / (self.k + base_ranked.get(item, base_beyond))
                + (1.0 - self.base_weight) / (self.k + pop_ranked.get(item, pop_beyond))
            )
        rated = set(matrix.indices[matrix.indptr[user_idx] : matrix.indptr[user_idx + 1]])
        return [
            (item, score)
            for item, score in sorted(scored.items(), key=lambda pair: -pair[1])
            if item not in rated
        ][:n]

    def predict(self, user_idx: int, item_idx: int) -> float:
        raise NotImplementedError("RankBlend is a ranking engine only")
