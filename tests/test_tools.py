import numpy as np
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
