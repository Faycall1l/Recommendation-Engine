import numpy as np
import pytest
import scipy.sparse as sp

from recagent.mf import ExplicitALS


def _low_rank_matrix(seed=0, n_users=8, n_items=10, rank=3):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, (n_users, rank))
    v = rng.normal(0.0, 1.0, (n_items, rank))
    return sp.csr_matrix(u @ v.T)


def _hand_matrix():
    return sp.csr_matrix([[5.0, 3.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0], [2.0, 0.0, 4.0, 0.0]])


def _train_rmse(mf, matrix):
    pred = mf.score_all()
    diff = pred - matrix.toarray()
    observed = matrix.nonzero()
    return float(np.sqrt(np.mean(diff[observed] ** 2)))


def test_als_recovers_low_rank_structure():
    matrix = _low_rank_matrix()
    mf = ExplicitALS(factors=3, iterations=60, regularization=0.01, seed=1).fit(matrix)
    assert _train_rmse(mf, matrix) < 0.1


def test_als_predict_matches_dot_product():
    matrix = _low_rank_matrix()
    mf = ExplicitALS(factors=3, iterations=10, seed=2).fit(matrix)
    assert mf.predict(3, 5) == pytest.approx(mf.user_factors[3] @ mf.item_factors[5])


def test_als_recommend_excludes_rated_and_sorts():
    matrix = _hand_matrix()
    mf = ExplicitALS(factors=4, iterations=15, seed=3).fit(matrix)
    out = mf.recommend(matrix, 0, n=2)
    assert len(out) == 2
    assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in out)
    assert {idx for idx, _ in out}.isdisjoint({0, 1})
    assert out[0][1] >= out[1][1]
