import numpy as np
import scipy.sparse as sp

from recagent.model import Recommender


def _matrix(n_users=40, n_items=60, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for u in range(n_users):
        k = rng.integers(5, 15)
        for _ in range(k):
            rows.append(u)
            cols.append(rng.integers(0, n_items))
    data = rng.uniform(1, 5, size=len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def test_fit_and_recommend():
    matrix = _matrix()
    rec = Recommender(factors=8, iterations=10).fit(matrix)

    user_idx = 3
    seen = set(matrix.getrow(user_idx).indices)
    top = rec.recommend(matrix, user_idx, n=10)

    assert len(top) == 10
    assert len({i for i, _ in top}) == 10  # distinct items
    assert all(i not in seen for i, _ in top)  # no already-seen items
    scores = [s for _, s in top]
    assert scores == sorted(scores, reverse=True)  # ranked by score


def test_similar_items_returns_itself_first():
    matrix = _matrix()
    rec = Recommender(factors=8, iterations=10).fit(matrix)
    ids, _ = zip(*rec.similar_items(5, n=5))
    assert ids[0] == 5


def test_roundtrip_save_load(tmp_path):
    matrix = _matrix()
    rec = Recommender(factors=8, iterations=5).fit(matrix)
    path = tmp_path / "model.npz"
    rec.save(path)

    loaded = Recommender.load(path)
    a = rec.recommend(matrix, 1, n=5)
    b = loaded.recommend(matrix, 1, n=5)
    assert a == b
