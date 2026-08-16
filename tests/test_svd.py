import numpy as np
import pytest
import scipy.sparse as sp

from recagent.svd import BiasedMF


def _low_rank_matrix(seed=0, n_users=8, n_items=10, rank=3):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, (n_users, rank))
    v = rng.normal(0.0, 1.0, (n_items, rank))
    biases = rng.normal(0.0, 0.5, size=n_users + n_items)
    mu = 3.0
    dense = mu + biases[:n_users][:, None] + biases[n_users:][None, :] + u @ v.T
    dense = np.clip(dense, 1.0, 5.0)
    return sp.csr_matrix(dense)


def _hand_matrix():
    return sp.csr_matrix([[5.0, 3.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0], [2.0, 0.0, 4.0, 0.0]])


def test_biased_mf_recovers_biased_low_rank_structure():
    matrix = _low_rank_matrix()
    mf = BiasedMF(factors=3, iterations=80, regularization=0.01, bias_shrinkage=100.0, seed=1).fit(matrix)
    dense = matrix.toarray()
    pred = mf.user_factors @ mf.item_factors.T + mf.mu + mf.user_bias[:, None] + mf.item_bias[None, :]
    observed = matrix.nonzero()
    rmse = float(np.sqrt(np.mean((pred - dense)[observed] ** 2)))
    # far below the global-mean baseline (~1.2 RMSE) proves the bias + factor
    # structure is actually recovered, not memorized
    assert rmse < 0.5


def test_biased_mf_predict_matches_formula():
    matrix = _low_rank_matrix()
    mf = BiasedMF(factors=3, iterations=10, seed=2).fit(matrix)
    expected = (
        mf.mu
        + mf.user_bias[3]
        + mf.item_bias[5]
        + mf.user_factors[3] @ mf.item_factors[5]
    )
    assert mf.predict(3, 5) == pytest.approx(float(expected))


def test_biased_mf_recommend_excludes_rated_and_sorts():
    matrix = _hand_matrix()
    mf = BiasedMF(factors=4, iterations=15, seed=3).fit(matrix)
    out = mf.recommend(matrix, 0, n=2)
    assert len(out) == 2
    assert {idx for idx, _ in out}.isdisjoint({0, 1})
    assert out[0][1] >= out[1][1]


def test_biased_mf_is_deterministic_with_same_seed():
    matrix = _hand_matrix()
    a = BiasedMF(factors=4, iterations=10, seed=5).fit(matrix)
    b = BiasedMF(factors=4, iterations=10, seed=5).fit(matrix)
    np.testing.assert_allclose(a.user_factors, b.user_factors)
    assert a.recommend(matrix, 0, n=2) == b.recommend(matrix, 0, n=2)


def test_biased_mf_roundtrip_save_load(tmp_path):
    matrix = _hand_matrix()
    original = BiasedMF(factors=4, iterations=10, regularization=0.2, seed=6).fit(matrix)
    path = tmp_path / "svd.npz"
    original.save(path)
    restored = BiasedMF.load(path)
    assert restored.factors == 4
    assert restored.regularization == pytest.approx(0.2)
    np.testing.assert_allclose(restored.user_factors, original.user_factors)
    np.testing.assert_allclose(restored.item_bias, original.item_bias)
    assert restored.recommend(matrix, 0, n=2) == original.recommend(matrix, 0, n=2)


def test_biased_mf_score_all_matches_per_user():
    matrix = _low_rank_matrix()
    mf = BiasedMF(factors=4, iterations=10, seed=7).fit(matrix)
    all_scores = mf.score_all()
    n_users, n_items = matrix.shape
    assert all_scores.shape == (n_users, n_items)
    for u in range(n_users):
        np.testing.assert_allclose(all_scores[u], mf._scores(u))
