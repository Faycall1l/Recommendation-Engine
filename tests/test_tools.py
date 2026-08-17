import numpy as np
import pytest
import scipy.sparse as sp

from recagent.model import Recommender
from recagent.tools import ToolRegistry


def build_state(n_users=10, n_items=20, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols, data = [], [], []
    for u in range(n_users):
        k = int(rng.integers(4, 8))
        for _ in range(k):
            rows.append(u)
            cols.append(int(rng.integers(0, n_items)))
            data.append(float(rng.integers(1, 6)))
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    user_ids = np.arange(101, 101 + n_users)
    item_ids = np.arange(1, 1 + n_items)
    genres = ["Action", "Drama", "Comedy"]
    items_meta = {
        int(iid): {"title": f"Movie {iid}", "genres": [genres[i % len(genres)]]}
        for i, iid in enumerate(item_ids)
    }
    uid_to_idx = {int(u): i for i, u in enumerate(user_ids)}
    iid_to_idx = {int(i): j for j, i in enumerate(item_ids)}
    model = Recommender(factors=8, iterations=5).fit(matrix)
    return {
        "model": model,
        "matrix": matrix,
        "uid_to_idx": uid_to_idx,
        "iid_to_idx": iid_to_idx,
        "user_ids": user_ids,
        "item_ids": item_ids,
        "items_meta": items_meta,
    }


def test_recommend_excludes_seen_and_returns_meta():
    state = build_state()
    registry = ToolRegistry(state)
    user_id = 101
    user_idx = state["uid_to_idx"][user_id]
    seen = set(state["matrix"].getrow(user_idx).indices)

    out = registry.recommend(user_id, n=10)

    assert out.user_id == user_id
    assert 0 < len(out.items) <= 10
    item_ids = [i.item_id for i in out.items]
    assert len(item_ids) == len(set(item_ids))
    idx = [state["iid_to_idx"][i] for i in item_ids]
    assert seen.isdisjoint(idx)  # already-seen items filtered by the engine
    for entry in out.items:
        assert entry.title.startswith("Movie")
        assert entry.genres


def test_item_info_and_user_profile():
    state = build_state()
    registry = ToolRegistry(state)
    info = registry.item_info(1)
    assert info.item_id == 1
    assert info.title == "Movie 1"
    assert info.rating_count >= 0

    profile = registry.user_profile(101, k=3)
    assert len(profile.items) <= 3
    ratings = [i.rating for i in profile.items]
    assert ratings == sorted(ratings, reverse=True)


def test_search_items_filters_by_genre():
    state = build_state()
    registry = ToolRegistry(state)
    out = registry.search_items("action", n=5)
    assert out.items, "expected at least one action movie"
    for entry in out.items:
        assert "Action" in entry.genres


def test_similar_items_excludes_seed():
    state = build_state()
    registry = ToolRegistry(state)
    out = registry.similar_items(1, n=5)
    assert len(out.items) <= 5
    assert all(i.item_id != 1 for i in out.items)
    assert all(i.score is not None for i in out.items)


def test_similar_users_excludes_self():
    state = build_state()
    registry = ToolRegistry(state)
    out = registry.similar_users(101, n=5)
    assert len(out.users) <= 5
    assert all(u.user_id != 101 for u in out.users)
    assert all(0.0 <= u.similarity <= 1.0 for u in out.users)


def test_filter_items_by_genre_and_rating():
    state = build_state()
    registry = ToolRegistry(state)
    out = registry.filter_items(genres=["Action"], min_rating=2.0, n=5)
    assert out.items
    for entry in out.items:
        assert "Action" in entry.genres
        assert entry.avg_rating is None or entry.avg_rating >= 2.0
    sorted_counts = [e.rating_count for e in out.items]
    assert sorted_counts == sorted(sorted_counts, reverse=True)


def test_trending_returns_popularity_prior():
    state = build_state()
    registry = ToolRegistry(state)
    out = registry.trending(n=5)
    assert 0 < len(out.items) <= 5
    for entry in out.items:
        assert entry.rating_count > 0
    counts = [e.score for e in out.items]
    assert counts == sorted(counts, reverse=True)


# ── RestrictedUnpickler tests ────────────────────────────────────────


def test_restricted_unpickler_allows_numpy():
    import io
    import pickle

    from recagent.state import RestrictedUnpickler

    obj = {"key": "value"}
    data = pickle.dumps(obj)
    loaded = RestrictedUnpickler(io.BytesIO(data)).load()
    assert loaded == obj


def test_rejected_unpickler_blocks_code_execution():
    """Verify that RestrictedUnpickler blocks os.system."""
    import io
    import pickle

    from recagent.state import RestrictedUnpickler

    data = b"cos\nsystem\n(S'echo pwned'\ntR."
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        RestrictedUnpickler(io.BytesIO(data)).load()


# ── ToolRegistry input validation tests ──────────────────────────────


def test_tool_registry_rejects_negative_user_id():
    state = build_state()
    registry = ToolRegistry(state)
    with pytest.raises((ValueError, KeyError)):
        registry.recommend(-1)


def test_tool_registry_rejects_negative_item_id():
    state = build_state()
    registry = ToolRegistry(state)
    with pytest.raises((ValueError, KeyError)):
        registry.similar_items(-1)


def test_tool_registry_rejects_zero_k():
    state = build_state()
    registry = ToolRegistry(state)
    with pytest.raises((ValueError, KeyError)):
        registry.recommend(101, n=0)


def test_tool_registry_rejects_k_over_max():
    state = build_state()
    registry = ToolRegistry(state)
    with pytest.raises((ValueError, KeyError)):
        registry.recommend(101, n=1001)


def test_tool_registry_rejects_non_int_user_id():
    state = build_state()
    registry = ToolRegistry(state)
    with pytest.raises((TypeError, KeyError)):
        registry.recommend("not_an_int")  # type: ignore[arg-type]
