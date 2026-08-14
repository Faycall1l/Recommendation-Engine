import pytest
import scipy.sparse as sp

from recagent.baselines import GlobalMean, ItemMean, MostPopular, RandomBaseline, UserMean
from recagent.cf import ItemBasedCF, UserBasedCF
from recagent.engines import (
    ENGINE_KINDS,
    RANKING_ENGINES,
    RATING_ENGINES,
    build_engine,
)
from recagent.mf import ExplicitALS
from recagent.model import Recommender


def _matrix():
    return sp.csr_matrix([[5.0, 3.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0], [2.0, 0.0, 4.0, 0.0]])


EXPECTED = {
    "als": Recommender,
    "user": UserBasedCF,
    "item": ItemBasedCF,
    "mf": ExplicitALS,
    "global-mean": GlobalMean,
    "user-mean": UserMean,
    "item-mean": ItemMean,
    "popular": MostPopular,
    "random": RandomBaseline,
}


@pytest.mark.parametrize("kind,cls", EXPECTED.items())
def test_build_engine_types(kind, cls):
    engine = build_engine(kind, _matrix(), factors=4, iterations=3)
    assert isinstance(engine, cls)


def test_build_engine_rejects_unknown():
    with pytest.raises(ValueError):
        build_engine("svd++", _matrix())


def test_build_engine_rating_kinds_predict():
    for kind in RATING_ENGINES:
        engine = build_engine(kind, _matrix(), factors=4, iterations=3)
        assert callable(engine.predict)
        score = engine.predict(0, 2)
        assert isinstance(score, float)


def test_build_engine_ranking_kinds_recommend():
    for kind in RANKING_ENGINES:
        engine = build_engine(kind, _matrix(), factors=4, iterations=3, seed=1)
        out = engine.recommend(_matrix(), 0, n=2)
        assert len(out) == 2
        assert {idx for idx, _ in out}.isdisjoint({0, 1})
        assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in out)


def test_engine_kind_groupings_are_complete():
    assert set(ENGINE_KINDS) == set(RATING_ENGINES) | set(RANKING_ENGINES)