import numpy as np
import pytest
import scipy.sparse as sp

from recagent.cf import CF_KINDS, ItemBasedCF, UserBasedCF


def _matrix() -> sp.csr_matrix:
    return sp.csr_matrix(
        [
            [5.0, 3.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 0.0],
            [2.0, 0.0, 4.0, 0.0],
        ]
    )


def test_cf_kinds():
    assert set(CF_KINDS) == {"als", "user", "item"}


def test_userbased_fit_means_and_centered():
    cf = UserBasedCF().fit(_matrix())
    np.testing.assert_allclose(cf.user_means, [4.0, 2.0, 3.0])
    expected = np.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(cf.centered.toarray(), expected)
    assert cf.matrix.format == "csr"


def test_userbased_similarity_pearson_hand_case():
    cf = UserBasedCF(min_sim=0.0).fit(_matrix())
    sim = cf.similarity
    # u0'=[1,-1,0,0], u1'=[0,-1,1,0] -> dot=1, /sqrt(2)*sqrt(2) = 0.5
    assert sim[0, 1] == pytest.approx(0.5)
    # u0' vs u2'=[-1,0,1,0] -> dot=-1 -> -0.5, floored to 0 at min_sim=0
    assert sim[0, 2] == 0.0
    # u1' vs u2' -> dot=1 -> 0.5
    assert sim[1, 2] == pytest.approx(0.5)
    np.testing.assert_allclose(np.diag(sim), 0.0)


def test_userbased_similarity_identical_users():
    matrix = sp.csr_matrix(
        [
            [5.0, 3.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 0.0],
            [5.0, 3.0, 0.0, 0.0],  # identical to user 0
        ]
    )
    cf = UserBasedCF().fit(matrix)
    assert cf.similarity[0, 2] == pytest.approx(1.0)


def test_userbased_similarity_min_sim_floor():
    cf = UserBasedCF(min_sim=0.6).fit(_matrix())
    assert (cf.similarity == 0.0).all()


def test_userbased_predict_hand_case():
    cf = UserBasedCF(min_sim=0.0).fit(_matrix())
    # item 2: deviations [0, 1, 1]; sim[0]=[0, .5, 0] -> 4 + 0.5/0.5 = 5.0
    assert cf.predict(0, 2) == pytest.approx(5.0)
    # item 3 is unrated everywhere -> prediction collapses to the user mean
    assert cf.predict(0, 3) == pytest.approx(4.0)
    assert cf.predict(2, 3) == pytest.approx(3.0)


def test_userbased_predict_falls_back_to_mean_without_neighbours():
    matrix = sp.csr_matrix([[5.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    cf = UserBasedCF().fit(matrix)
    assert cf.predict(0, 1) == pytest.approx(5.0)


def test_userbased_recommend_hand_case():
    matrix = _matrix()
    cf = UserBasedCF(min_sim=0.0).fit(matrix)
    # user 0: scores [4,3,5,4], rated {0,1} -> unseen 2 (5.0), 3 (4.0)
    assert cf.recommend(matrix, 0, n=2) == [(2, 5.0), (3, 4.0)]
    # user 2: scores [3,2,4,3], rated {0,2} -> unseen 3 (3.0), 1 (2.0)
    assert cf.recommend(matrix, 2, n=2) == [(3, 3.0), (1, 2.0)]


def test_userbased_recommend_excludes_rated_and_caps_n():
    matrix = _matrix()
    cf = UserBasedCF(min_sim=0.0).fit(matrix)
    out = cf.recommend(matrix, 0, n=10)
    rated = {0, 1}
    ids = [idx for idx, _ in out]
    assert rated.isdisjoint(ids)
    assert len(out) == 2  # only two unseen items exist
    assert all(isinstance(score, float) for _, score in out)


def test_userbased_score_all_batch_matches_predict():
    matrix = _matrix()
    cf = UserBasedCF(min_sim=0.0).fit(matrix)
    all_scores = cf.score_all()
    assert all_scores.shape == matrix.shape
    np.testing.assert_allclose(all_scores[0], [4.0, 3.0, 5.0, 4.0])
    np.testing.assert_allclose(all_scores[2], [3.0, 2.0, 4.0, 3.0])
    for u in range(3):
        for j in range(4):
            assert all_scores[u, j] == pytest.approx(cf.predict(u, j))


def test_userbased_score_all_no_neighbours_is_user_mean():
    matrix = sp.csr_matrix([[5.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    cf = UserBasedCF().fit(matrix)
    all_scores = cf.score_all()
    np.testing.assert_allclose(all_scores[0], [5.0, 5.0, 5.0])
    assert np.isfinite(all_scores).all()


def test_userbased_roundtrip_save_load(tmp_path):
    matrix = _matrix()
    original = UserBasedCF(min_sim=0.1).fit(matrix)
    path = tmp_path / "model_cf.npz"
    original.save(path)

    restored = UserBasedCF.load(path)
    assert restored.min_sim == pytest.approx(0.1)
    np.testing.assert_allclose(restored.user_means, original.user_means)
    np.testing.assert_allclose(restored.similarity, original.similarity)
    np.testing.assert_allclose(restored.score_all(), original.score_all())
    np.testing.assert_allclose(restored.centered.toarray(), original.centered.toarray())
    assert restored.recommend(matrix, 0, n=2) == original.recommend(matrix, 0, n=2)


def test_itembased_fit_means_and_similarity():
    cf = ItemBasedCF(min_sim=0.0).fit(_matrix())
    # item means: (5+2)/2=3.5, (3+1)/2=2, (3+4)/2=3.5, unrated=0
    np.testing.assert_allclose(cf.item_means, [3.5, 2.0, 3.5, 0.0])
    sim = cf.similarity
    assert sim[0, 1] == pytest.approx(0.5)
    assert sim[0, 2] == 0.0  # -0.5 floored at min_sim=0
    assert sim[1, 2] == pytest.approx(0.5)
    np.testing.assert_allclose(np.diag(sim), 0.0)
    assert sim.shape == (4, 4)


def test_itembased_predict_hand_case():
    cf = ItemBasedCF(min_sim=0.0).fit(_matrix())
    # user 0 rates item1=3; item2 similar to item1 (0.5) -> (0.5*3)/0.5 = 3.0
    assert cf.predict(0, 2) == pytest.approx(3.0)
    # item3 is similar to nothing -> falls back to the user mean (4.0)
    assert cf.predict(0, 3) == pytest.approx(4.0)


def test_itembased_recommend_hand_case():
    matrix = _matrix()
    cf = ItemBasedCF(min_sim=0.0).fit(matrix)
    # user 0 rated {0:5, 1:3}; preds item0=3.0, item1=5.0, item2=3.0, item3=4.0
    # unseen: item2 (3.0), item3 (4.0) -> item3 first
    out = cf.recommend(matrix, 0, n=2)
    assert [idx for idx, _ in out] == [3, 2]
    assert out[0][1] == pytest.approx(4.0)
    assert out[1][1] == pytest.approx(3.0)


def test_itembased_score_all_matches_predict():
    cf = ItemBasedCF(min_sim=0.0).fit(_matrix())
    all_scores = cf.score_all()
    assert all_scores.shape == cf.matrix.shape
    assert all_scores[0, 2] == pytest.approx(cf.predict(0, 2))
    assert all_scores[0, 3] == pytest.approx(cf.predict(0, 3))
    assert all_scores[2, 0] == pytest.approx(cf.predict(2, 0))
    assert np.isfinite(all_scores).all()


def test_itembased_roundtrip_save_load(tmp_path):
    matrix = _matrix()
    original = ItemBasedCF(min_sim=0.1).fit(matrix)
    path = tmp_path / "model_cf.npz"
    original.save(path)

    restored = ItemBasedCF.load(path)
    assert restored.min_sim == pytest.approx(0.1)
    np.testing.assert_allclose(restored.similarity, original.similarity)
    np.testing.assert_allclose(restored.user_means, original.user_means)
    np.testing.assert_allclose(restored.item_means, original.item_means)
    assert restored.predict(0, 2) == pytest.approx(original.predict(0, 2))
    assert restored.recommend(matrix, 0, n=2) == original.recommend(matrix, 0, n=2)
    np.testing.assert_allclose(restored.score_all(), original.score_all())
