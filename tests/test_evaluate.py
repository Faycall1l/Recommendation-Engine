import numpy as np
import pytest
import scipy.sparse as sp

from recagent.cf import build_cf
from recagent.evaluate import (
    aligned_rank_arrays,
    cf_baseline,
    cv_rating_eval_from_arrays,
    genre_precision,
    head_item_ids,
    hits_from_ranks,
    loo_ranking_eval_from_arrays,
    mean_metrics,
    paired_bootstrap,
    rating_metrics,
)
from tests.test_tools import build_state


def _user_state():
    matrix = sp.csr_matrix([[5.0, 3.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0], [2.0, 0.0, 4.0, 0.0]])
    return {
        "model": build_cf("user", matrix),
        "matrix": matrix,
        "uid_to_idx": {1: 0, 2: 1, 3: 2},
        "iid_to_idx": {11: 0, 12: 1, 13: 2, 14: 3},
        "user_ids": [1, 2, 3],
        "item_ids": [11, 12, 13, 14],
        "items_meta": {},
        "cf_kind": "user",
    }


def test_mean_metrics_hits_and_ranking():
    ranked = {1: [10, 20, 30], 2: [10, 20, 30], 3: [10, 20, 30]}
    test_items = {1: 30, 2: 10, 3: 999}
    metrics = mean_metrics(ranked, test_items, ks=(1, 3))

    assert metrics["n_users"] == 3
    # user 3 misses everywhere
    assert metrics["hr"]["1"] == round(1 / 3, 4)
    assert metrics["hr"]["3"] == round(2 / 3, 4)
    # user1 target at rank 3 (gain 1/2), user2 at rank 1 (gain 1)
    expected_ndcg3 = round((1 / np.log2(4) + 1 / np.log2(2)) / 3, 4)
    assert metrics["ndcg"]["3"] == expected_ndcg3


def test_mean_metrics_empty_rankings_do_not_crash():
    metrics = mean_metrics({1: [], 2: [5]}, {1: 7, 2: 5}, ks=(1, 5))
    assert metrics["n_users"] == 1
    assert metrics["hr"]["5"] == 1.0


def test_mean_metrics_extended_ranking_metrics():
    ranked = {1: [10, 20, 30], 2: [10, 20, 30], 3: [10, 20, 30]}
    test_items = {1: 30, 2: 10, 3: 999}
    m = mean_metrics(ranked, test_items, ks=(1, 3))
    n = 3
    # recall == hr for single-relevant protocol
    assert m["recall"]["3"] == m["hr"]["3"] == round(2 / 3, 4)
    # precision@k = hits / (n * k)
    assert m["precision"]["1"] == round(1 / n, 4)
    assert m["precision"]["3"] == round(2 / (n * 3), 4)
    # MAP: user1 1/3 (rank3), user2 1/1 (rank1)
    assert m["map"]["3"] == round((1 / 3 + 1) / n, 4)
    # MRR: same two users
    assert m["mrr"] == round((1 / 3 + 1) / n, 4)


def test_cf_baseline_runs_and_is_deterministic():
    state = build_state(n_users=12, n_items=25)
    test_items = {101: 1, 102: 2, 103: 3}
    a = cf_baseline(state, test_items, ks=(1, 5, 10))
    b = cf_baseline(state, test_items, ks=(1, 5, 10))
    assert a == b
    assert a["n_users"] <= 3


def test_cf_baseline_user_engine_rankings():
    state = _user_state()
    test_items = {1: 13, 2: 12, 3: 11}
    # user 1 (idx 0): unseen items 13 (5.0) and 14 (4.0) -> 13 at rank 1
    metrics = cf_baseline(state, test_items, kind="user", ks=(1, 3, 5, 10))
    assert metrics["kind"] == "user"
    assert metrics["n_users"] == 3
    assert metrics["hr"]["1"] == round(1 / 3, 4)
    # user 2 (idx 1) target 12 is rated -> missed; user 3 (idx 2) target 11 rated -> missed
    assert metrics["hr"]["10"] == round(1 / 3, 4)


def test_cf_baseline_item_engine_rankings():
    state = _user_state()
    test_items = {1: 13, 2: 12, 3: 11}
    # item engine ranks user1's unseen as 14 (4.0) then 13 (3.0) -> 13 at rank 2
    metrics = cf_baseline(state, test_items, kind="item", ks=(1, 3, 5, 10))
    assert metrics["kind"] == "item"
    assert metrics["hr"]["1"] == 0.0
    assert metrics["hr"]["3"] == round(1 / 3, 4)


def test_cf_baseline_rejects_unknown_kind():
    with pytest.raises(ValueError):
        cf_baseline(_user_state(), {1: 13}, kind="bogus")


def test_rating_metrics_rmse_and_mae():
    actual = np.array([3.0, 4.0, 5.0])
    predicted = np.array([2.0, 4.0, 6.0])
    m = rating_metrics(actual, predicted)
    assert m["n"] == 3
    assert m["rmse"] == pytest.approx(np.sqrt(2 / 3), abs=1e-4)
    assert m["mae"] == pytest.approx(2 / 3, abs=1e-4)


def test_rating_metrics_empty():
    assert rating_metrics([], []) == {"rmse": 0.0, "mae": 0.0, "n": 0}


def _low_rank_ratings(seed=0, n_users=30, n_items=40, rank=4):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, (n_users, rank))
    v = rng.normal(0.0, 1.0, (n_items, rank))
    scores = u @ v.T
    scores = (scores - scores.min()) / (scores.max() - scores.min()) * 4 + 1
    users = np.repeat(np.arange(1, n_users + 1), n_items)
    items = np.tile(np.arange(1, n_items + 1), n_users)
    return users, items, scores.ravel()


def test_cv_rating_eval_runs_and_orders_engines():
    users, items, ratings = _low_rank_ratings()
    results = cv_rating_eval_from_arrays(
        users, items, ratings, kinds=("mf", "user", "global-mean"), k=3, seed=1, factors=4, iterations=15
    )
    assert set(results) == {"mf", "user", "global-mean"}
    for metrics in results.values():
        assert len(metrics["per_fold"]) == 3
        assert 0 <= metrics["rmse"] <= 5
        assert 0 <= metrics["mae"] <= 5
    # matrix factorization recovers the low-rank signal far better than the mean
    assert results["mf"]["rmse"] < results["global-mean"]["rmse"]
    assert results["user"]["rmse"] < results["global-mean"]["rmse"]


def test_cv_rating_eval_rejects_ranking_only_engines():
    users, items, ratings = _low_rank_ratings()
    with pytest.raises(ValueError):
        cv_rating_eval_from_arrays(users, items, ratings, kinds=("als",))


def test_cv_rating_eval_engine_kwargs_per_engine_config():
    users, items, ratings = _low_rank_ratings()
    results = cv_rating_eval_from_arrays(
        users,
        items,
        ratings,
        kinds=("mf",),
        k=3,
        seed=1,
        engine_kwargs={"mf": {"factors": 3, "iterations": 10, "regularization": 0.5}},
    )
    # per-engine config is respected AND recorded for provenance
    assert results["mf"]["config"] == {"factors": 3, "iterations": 10, "regularization": 0.5}


def _taste_group_ratings(seed=0):
    """24 users in 3 taste groups; neighbours share the group's liked items."""
    rng = np.random.default_rng(seed)
    users, items, ratings = [], [], []
    groups = [(0, range(6), [13, 14]), (8, range(6, 12), [0, 1]), (16, range(12, 16), [2, 3])]
    for u_start, liked, noise in groups:
        liked = list(liked)
        for offset in range(8):
            u = u_start + offset + 1
            for i in liked:
                users.append(u)
                items.append(i + 1)
                ratings.append(float(rng.integers(4, 6)))
            for i in noise:  # noise items are outside the liked set, so no dup (u,i)
                users.append(u)
                items.append(i + 1)
                ratings.append(1.0)
    return np.asarray(users), np.asarray(items), np.asarray(ratings)


def test_loo_ranking_eval_all_engines_and_baseline_bounds():
    users, items, ratings = _taste_group_ratings()
    results = loo_ranking_eval_from_arrays(
        users, items, ratings, kinds=("user", "popular", "random"), seed=1
    )
    assert set(results) == {"user", "popular", "random"}
    for kind, metrics in results.items():
        assert metrics["kind"] == kind
        assert "mrr" in metrics
        assert set(metrics["map"]) == set(metrics["hr"])
    # structured taste means a real filter beats random, even against popularity
    # (item counts are near-uniform on this toy set, so popular is ~uniform too)
    assert results["user"]["hr"]["5"] > results["random"]["hr"]["5"]


def test_loo_ranking_eval_user_sample():
    users, items, ratings = _taste_group_ratings()
    results = loo_ranking_eval_from_arrays(
        users, items, ratings, kinds=("user",), seed=1, user_sample=8
    )
    assert results["user"]["n_users"] == 8


def test_loo_ranking_eval_engine_kwargs():
    users, items, ratings = _taste_group_ratings()
    results = loo_ranking_eval_from_arrays(
        users,
        items,
        ratings,
        kinds=("mf", "random"),
        seed=1,
        engine_kwargs={"mf": {"factors": 3, "iterations": 10, "regularization": 1.0}},
    )
    assert set(results) == {"mf", "random"}
    assert "mrr" in results["mf"]


def test_loo_ranking_eval_rejects_rating_only_engines():
    users, items, ratings = _taste_group_ratings()
    with pytest.raises(ValueError):
        loo_ranking_eval_from_arrays(users, items, ratings, kinds=("global-mean",))


def test_hits_from_ranks():
    ranks = {1: 3, 2: 1, 3: 0, 4: 5}
    assert hits_from_ranks(ranks, 5).tolist() == [1, 1, 0, 1]
    assert hits_from_ranks(ranks, 2).tolist() == [0, 1, 0, 0]


def test_aligned_rank_arrays_pairs_by_user_not_position():
    # int keys in one map, str keys in the other; per-user rank depends on the
    # user id, so a misaligned (position-based) pairing would be detectable
    baseline = {str(u): (1 if u % 2 == 0 else 0) for u in range(1, 11)}
    agent = {u: (1 if u % 2 == 1 else 0) for u in range(1, 11)}
    ha, hb, _ma, _mb, uids = aligned_rank_arrays(baseline, agent, 5)
    assert uids == list(range(1, 11))
    # index i must reference uids[i] for BOTH maps
    for i, u in enumerate(uids):
        assert ha[i] == float(baseline[str(u)] > 0)
        assert hb[i] == float(agent[u] > 0)
    # anti-correlated pairing: agent hits exactly where baseline misses
    assert list(hb) == [1 - float(x) for x in ha]


def test_paired_bootstrap_finds_separation():
    a = np.ones(40)
    b = np.zeros(40)
    out = paired_bootstrap(a, b, n_boot=1000, seed=1)
    assert out["mean_diff"] == 1.0
    assert out["ci_lo"] > 0.9 and out["ci_hi"] <= 1.0
    assert out["p_value"] < 0.01


def test_paired_bootstrap_null_is_insignificant():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=50)
    out = paired_bootstrap(x, x, n_boot=1000, seed=1)
    assert out["mean_diff"] == 0.0
    assert out["p_value"] == 1.0
    assert out["ci_lo"] <= 0 <= out["ci_hi"]


def test_paired_bootstrap_is_deterministic():
    rng = np.random.default_rng(0)
    a = rng.uniform(size=30)
    b = rng.uniform(size=30)
    first = paired_bootstrap(a, b, n_boot=500, seed=7)
    second = paired_bootstrap(a, b, n_boot=500, seed=7)
    assert first == second


def test_paired_bootstrap_validates_input():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def test_paired_bootstrap_cohens_d():
    rng = np.random.default_rng(42)
    a = rng.normal(1.0, 1.0, size=40)
    b = rng.normal(0.0, 1.0, size=40)
    out = paired_bootstrap(a, b, n_boot=100, seed=1)
    # Cohen's d is mean_diff / pooled_std, should be positive since a > b
    assert out["cohens_d"] > 0


def test_paired_bootstrap_cohens_d_zero_when_identical():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=50)
    out = paired_bootstrap(x, x, n_boot=100, seed=1)
    assert out["cohens_d"] == 0.0


def test_genre_precision_is_case_insensitive():
    share = {"Film-Noir": 1.0, "Sci-Fi": 0.5}
    assert genre_precision(share, "film-noir") == 1.0
    assert genre_precision(share, "sci-fi") == 0.5
    assert genre_precision(share, "comedy") == 0.0


def test_head_item_ids_marks_popular_items():
    items = np.asarray([10, 10, 10, 10, 10, 20, 30, 40, 50])
    assert head_item_ids(items, 0.25) == {10}  # 1 of 4 distinct items
    assert head_item_ids(items, 0.0) == set()
    assert head_item_ids(items, 0.75) == {10, 20, 30, 40}  # top 4


def test_loo_exclude_head_removes_popular_targets():
    # item 10 is overwhelmingly the most popular; some LOO targets are 10
    users = np.asarray(
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    )
    items = np.asarray(
        [10, 10, 10, 20, 30, 10, 40, 50, 60, 70, 10, 80, 90, 100, 110]
    )
    ratings = np.full(len(users), 4.0)
    full = loo_ranking_eval_from_arrays(users, items, ratings, kinds=("popular",))
    tail = loo_ranking_eval_from_arrays(
        users, items, ratings, kinds=("popular",), exclude_head=0.1
    )
    assert tail["popular"]["n_users"] < full["popular"]["n_users"]
    assert tail["popular"]["exclude_head"] == 0.1


def test_loo_exclude_head_validates_range():
    users, items, ratings = _taste_group_ratings()
    with pytest.raises(ValueError):
        loo_ranking_eval_from_arrays(users, items, ratings, exclude_head=1.0)
    with pytest.raises(ValueError):
        loo_ranking_eval_from_arrays(users, items, ratings, exclude_head=-0.5)
