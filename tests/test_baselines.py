import pytest
import scipy.sparse as sp

from recagent.baselines import GlobalMean, ItemMean, MostPopular, RandomBaseline, UserMean


def _matrix():
    return sp.csr_matrix([[5.0, 3.0, 0.0], [0.0, 1.0, 3.0]])


def test_global_mean():
    b = GlobalMean().fit(_matrix())
    assert b.predict(0, 0) == pytest.approx(12.0 / 4)
    with pytest.raises(NotImplementedError):
        b.recommend(_matrix(), 0)


def test_user_mean():
    b = UserMean().fit(_matrix())
    assert b.predict(0, 2) == pytest.approx(4.0)
    assert b.predict(1, 0) == pytest.approx(2.0)


def test_item_mean():
    b = ItemMean().fit(_matrix())
    assert b.predict(1, 0) == pytest.approx(5.0)
    assert b.predict(0, 1) == pytest.approx(2.0)


def test_most_popular_rankings():
    b = MostPopular().fit(_matrix())
    # counts: item0=1, item1=2, item2=1 -> order [1,0,2]; user0 rated {0,1}
    assert b.recommend(_matrix(), 0, n=2) == [(2, 1.0)]


def test_random_baseline_deterministic_and_excludes_rated():
    matrix = _matrix()
    a = RandomBaseline(seed=3).fit(matrix)
    b = RandomBaseline(seed=3).fit(matrix)
    assert a.recommend(matrix, 0, n=2) == b.recommend(matrix, 0, n=2)
    out = a.recommend(matrix, 0, n=2)
    assert [idx for idx, _ in out] == [2]  # only item 2 is unrated by user 0
