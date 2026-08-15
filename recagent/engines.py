"""Single entry point for building any evaluation engine by name.

Unifies the from-scratch neighbourhood methods, the from-scratch matrix
factorizer, the ALS wrapper and the trivial baselines behind one factory so
protocols can treat every engine uniformly.
"""

from __future__ import annotations

import scipy.sparse as sp

from recagent.baselines import GlobalMean, ItemMean, MostPopular, RandomBaseline, UserMean
from recagent.blend import RankBlend
from recagent.cf import ItemBasedCF, UserBasedCF
from recagent.mf import ExplicitALS
from recagent.svd import BiasedMF

ALS = "als"
USER = "user"
ITEM = "item"
MF = "mf"
SVD = "svd"
BLEND = "blend"
GLOBAL_MEAN = "global-mean"
USER_MEAN = "user-mean"
ITEM_MEAN = "item-mean"
POPULAR = "popular"
RANDOM = "random"

ENGINE_KINDS = (
    ALS,
    USER,
    ITEM,
    MF,
    SVD,
    BLEND,
    GLOBAL_MEAN,
    USER_MEAN,
    ITEM_MEAN,
    POPULAR,
    RANDOM,
)

# ALS and BLEND rank but have no calibrated explicit-rating predictor
# (blend delegates predict() to nothing), so both are excluded from the
# rating protocol.
RATING_ENGINES = (USER, ITEM, MF, SVD, GLOBAL_MEAN, USER_MEAN, ITEM_MEAN)
RANKING_ENGINES = (ALS, USER, ITEM, MF, SVD, BLEND, POPULAR, RANDOM)


def build_engine(
    kind: str,
    matrix: sp.csr_matrix,
    *,
    seed: int = 42,
    factors: int = 64,
    iterations: int = 20,
    regularization: float = 0.1,
    topk: int | None = None,
    base_kind: str = ALS,
    base_weight: float = 0.6,
    top_k: int = 200,
    rrf_k: int = 60,
):
    """Fit and return the engine named by ``kind`` (see :data:`ENGINE_KINDS`).

    ``topk`` (memory-based engines only) keeps a sparse top-k neighbourhood
    instead of the full dense similarity — required at large user/item counts.
    """
    kind = kind.lower()
    if kind not in ENGINE_KINDS:
        raise ValueError(f"unknown engine kind {kind!r}; expected one of {ENGINE_KINDS}")
    if kind == ALS:
        from recagent.model import Recommender

        return Recommender(factors=factors, iterations=iterations, regularization=regularization).fit(matrix)
    if kind == USER:
        return UserBasedCF(topk=topk).fit(matrix)
    if kind == ITEM:
        return ItemBasedCF(topk=topk).fit(matrix)
    if kind == MF:
        return ExplicitALS(
            factors=factors, iterations=iterations, regularization=regularization, seed=seed
        ).fit(matrix)
    if kind == SVD:
        return BiasedMF(
            factors=factors, iterations=iterations, regularization=regularization, seed=seed
        ).fit(matrix)
    if kind == BLEND:
        return RankBlend(
            base_kind=base_kind,
            base_weight=base_weight,
            top_k=top_k,
            k=rrf_k,
            seed=seed,
            factors=factors,
            iterations=iterations,
            regularization=regularization,
        ).fit(matrix)
    if kind == GLOBAL_MEAN:
        return GlobalMean().fit(matrix)
    if kind == USER_MEAN:
        return UserMean().fit(matrix)
    if kind == ITEM_MEAN:
        return ItemMean().fit(matrix)
    if kind == POPULAR:
        return MostPopular().fit(matrix)
    if kind == RANDOM:
        return RandomBaseline(seed=seed).fit(matrix)
    raise AssertionError(f"unhandled kind {kind!r}")
