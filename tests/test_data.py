import numpy as np

from recagent.data import encode, leave_one_out


def test_encode_shapes_and_values():
    users = np.array([1, 1, 2, 2, 3])
    items = np.array([10, 11, 10, 12, 13])
    ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    matrix, uid_to_idx, iid_to_idx, _user_ids, _item_ids = encode(users, items, ratings)

    assert matrix.shape == (3, 4)
    assert uid_to_idx == {1: 0, 2: 1, 3: 2}
    assert len(iid_to_idx) == 4
    # (user 2, item 10) rating 3.0 lands correctly
    assert matrix[uid_to_idx[2], iid_to_idx[10]] == 3.0
    assert matrix[uid_to_idx[1], iid_to_idx[10]] == 1.0
    assert matrix.getnnz() == 5


def test_encode_pinned_index_space():
    users = np.array([1, 2])
    items = np.array([10, 20])
    ratings = np.array([1.0, 2.0])
    _, _, _, _, item_ids = encode(users, items, ratings, item_ids=np.array([10, 20, 30]))
    assert list(item_ids) == [10, 20, 30]


def test_leave_one_out_disjoint_and_sized():
    # one unique item per user-interaction to avoid duplicate (user, item) pairs
    rng = np.random.default_rng(0)
    users = np.repeat(np.arange(1, 11), 10)
    items = np.concatenate(
        [rng.choice(100, size=10, replace=False) + 1000 * u for u in range(1, 11)]
    )
    ratings = np.ones(100)

    (tr_u, tr_i, _tr_r), (te_u, te_i) = leave_one_out(users, items, ratings, seed=7)

    # every eligible user loses exactly one interaction
    assert set(tr_u) == set(range(1, 11))
    assert len(te_u) == 10
    assert all(u in tr_u for u in te_u)
    # held-out interactions never appear in the train arrays
    held_pairs = set(zip(te_u, te_i))
    train_pairs = set(zip(tr_u, tr_i))
    assert held_pairs.isdisjoint(train_pairs)


def test_leave_one_out_skips_cold_users():
    users = np.array([1, 1, 1, 1, 1, 2, 2])
    items = np.arange(7)
    ratings = np.ones(7)
    _, (te_u, _te_i) = leave_one_out(users, items, ratings, min_interactions=5)
    # user 2 has too few interactions and must be excluded
    assert 2 not in te_u
