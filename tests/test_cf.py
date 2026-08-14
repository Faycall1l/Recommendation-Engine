import numpy as np
import pytest
import scipy.sparse as sp

from recagent.cf import CF_KINDS, UserBasedCF


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
