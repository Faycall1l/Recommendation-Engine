import numpy as np
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
