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


def test_als_is_deterministic_with_same_seed():
    matrix = _hand_matrix()
    a = ExplicitALS(factors=4, iterations=10, seed=5).fit(matrix)
    b = ExplicitALS(factors=4, iterations=10, seed=5).fit(matrix)
    np.testing.assert_allclose(a.user_factors, b.user_factors)
    np.testing.assert_allclose(a.item_factors, b.item_factors)
    assert a.recommend(matrix, 0, n=2) == b.recommend(matrix, 0, n=2)


def test_als_roundtrip_save_load(tmp_path):
    matrix = _hand_matrix()
    original = ExplicitALS(factors=4, iterations=10, regularization=0.2, seed=6).fit(matrix)
    path = tmp_path / "mf.npz"
    original.save(path)
    restored = ExplicitALS.load(path)
    assert restored.factors == 4
    assert restored.iterations == 10
    assert restored.regularization == pytest.approx(0.2)
    np.testing.assert_allclose(restored.user_factors, original.user_factors)
    np.testing.assert_allclose(restored.item_factors, original.item_factors)
    assert restored.recommend(matrix, 0, n=2) == original.recommend(matrix, 0, n=2)


def test_als_score_all_matches_per_user():
    matrix = _low_rank_matrix()
    mf = ExplicitALS(factors=4, iterations=10, seed=7).fit(matrix)
    all_scores = mf.score_all()
    n_users, n_items = matrix.shape
    assert all_scores.shape == (n_users, n_items)
    for u in range(n_users):
        np.testing.assert_allclose(
            all_scores[u], mf.user_factors[u] @ mf.item_factors.T
        )
