import scipy.sparse as sp

from recagent.blend import RankBlend


def _hand_matrix():
    return sp.csr_matrix([[5.0, 3.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0], [2.0, 0.0, 4.0, 0.0]])


def test_rank_blend_fit_and_recommend_bounds():
    matrix = _hand_matrix()
    blend = RankBlend(base_kind="user", base_weight=0.5, top_k=10, k=5, seed=1).fit(matrix)
    out = blend.recommend(matrix, 0, n=2)
    assert len(out) == 2
    assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in out)
    # rated items are never recommended
    assert {idx for idx, _ in out}.isdisjoint({0, 1})
    assert out[0][1] >= out[1][1]


def test_rank_blend_weight_extremes():
    matrix = _hand_matrix()
    # pure-popularity blend must rank by global rating counts
    blend = RankBlend(base_kind="user", base_weight=0.0, top_k=10, k=1, seed=1).fit(matrix)
    top = blend.recommend(matrix, 2, n=3)
    counts = matrix.getnnz(axis=0)
    ranks = [counts[item_idx] for item_idx, _ in top]
    assert ranks == sorted(ranks, reverse=True)
    # and must never recommend rated items
    assert {idx for idx, _ in top}.isdisjoint({0, 2})


def test_rank_blend_matches_pure_base_at_weight_one():
    matrix = _hand_matrix()
    from recagent.cf import UserBasedCF

    base = UserBasedCF().fit(matrix)
    blend = RankBlend(base_kind="user", base_weight=1.0, top_k=10, k=60, seed=1).fit(matrix)
    # RRF rescales scores, so compare the item ordering only
    assert [idx for idx, _ in blend.recommend(matrix, 0, n=2)] == [idx for idx, _ in base.recommend(matrix, 0, n=2)]
