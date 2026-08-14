import numpy as np

from recagent.evaluate import cf_baseline, mean_metrics
from tests.test_tools import build_state


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
