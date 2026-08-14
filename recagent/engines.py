"""Single entry point for building any evaluation engine by name.

Unifies the from-scratch neighbourhood methods, the from-scratch matrix
factorizer, the ALS wrapper and the trivial baselines behind one factory so
protocols can treat every engine uniformly.
"""

from __future__ import annotations

import scipy.sparse as sp

from recagent.baselines import GlobalMean, ItemMean, MostPopular, RandomBaseline, UserMean
from recagent.cf import ItemBasedCF, UserBasedCF
from recagent.mf import ExplicitALS

ALS = "als"
USER = "user"
ITEM = "item"
MF = "mf"
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
    GLOBAL_MEAN,
    USER_MEAN,
    ITEM_MEAN,
    POPULAR,
    RANDOM,
)

# ALS is the implicit-feedback wrapper: it ranks but has no calibrated
# explicit-rating predictor, so it is excluded from the rating protocol.
RATING_ENGINES = (USER, ITEM, MF, GLOBAL_MEAN, USER_MEAN, ITEM_MEAN)
RANKING_ENGINES = (ALS, USER, ITEM, MF, POPULAR, RANDOM)


def build_engine(
    kind: str,
    matrix: sp.csr_matrix,
    *,
    seed: int = 42,
    factors: int = 64,
    iterations: int = 20,
    regularization: float = 0.1,
    topk: int | None = None,
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
