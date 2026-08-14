import numpy as np
import pytest
import scipy.sparse as sp

from recagent.cf import build_cf
from recagent.evaluate import cf_baseline, mean_metrics
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
