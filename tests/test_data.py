import numpy as np
import pytest

from recagent.data import (
    _year_from_title,
    encode,
    leave_one_out,
    load_items_20m,
    load_ratings_20m,
    loaders,
    split_ratings,
)


def test_load_ratings_20m(tmp_path):
    ratings = tmp_path / "ratings.csv"
    ratings.write_text(
        "userId,movieId,rating,timestamp\n"
        "1,296,5.0,1111111111\n"
        "1,296,5.0,1111111112\n"
        "2,1,3.5,1111111113\n"
    )
    users, items, values = load_ratings_20m(tmp_path)
    assert users.tolist() == [1, 1, 2]
    assert items.tolist() == [296, 296, 1]
    assert values.tolist() == [5.0, 5.0, 3.5]


def test_load_items_20m_parses_quoted_titles(tmp_path):
    movies = tmp_path / "movies.csv"
    movies.write_text(
        "movieId,title,genres\n"
        "1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy\n"
        '356,"Where the Boys Are, \'84 (1984)",Comedy\n'
        "10,Seven Samurai (1954),Action|Adventure|Drama\n"
    )
    items = load_items_20m(tmp_path)
    assert items[1]["title"] == "Toy Story (1995)"
    assert items[1]["year"] == 1995
    assert items[1]["genres"] == ["Adventure", "Animation", "Children", "Comedy", "Fantasy"]
    assert items[356]["title"] == "Where the Boys Are, '84 (1984)"
    assert items[356]["year"] == 1984
    assert items[356]["genres"] == ["Comedy"]
    assert items[10]["year"] == 1954
    assert items[10]["genres"] == ["Action", "Adventure", "Drama"]


def test_year_from_title():
    assert _year_from_title("Toy Story (1995)") == 1995
    assert _year_from_title("Seven Samurai (1954)") == 1954
    assert _year_from_title("A Movie") is None
    assert _year_from_title("No Year Here (x)") is None


def test_loaders_dispatch():
    fetch, load_ratings, load_items = loaders("ml-100k")
    assert fetch.__name__ == "fetch_movielens"
    assert load_ratings.__name__ == "load_ratings"
    assert load_items.__name__ == "load_items"
    fetch20, load20, items20 = loaders("ml-20m")
    assert fetch20.__name__ == "fetch_movielens_20m"
    assert load20.__name__ == "load_ratings_20m"
    assert items20.__name__ == "load_items_20m"
    with pytest.raises(ValueError):
        loaders("ml-9000")


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


def test_split_ratings_partitions_all_triples():
    users = np.arange(1, 11).repeat(10)
    items = np.tile(np.arange(100, 110), 10)
    ratings = np.ones(100)
    folds = split_ratings(users, items, ratings, k=5, seed=0)

    assert len(folds) == 5
    total = sum(len(te_r) for _, (_, _, te_r) in folds)
    assert total == 100
    for (tr_u, _tr_i, _tr_r), (_te_u, _te_i, _te_r) in folds:
        assert len(tr_u) + len(_te_u) == 100
        assert len(_te_u) == 20  # balanced 100 / 5


def test_split_ratings_train_coverage_guard():
    users = np.array([1, 1, 1, 2])
    items = np.array([10, 11, 12, 20])
    ratings = np.ones(4)
    # user 2 has a single interaction; with k=2 one of its slots may be split out
    for (tr_u, _tr_i, _tr_r), (te_u, _te_i, _te_r) in split_ratings(users, items, ratings, k=2, seed=1):
        train_has = set(tr_u)
        for u in te_u:
            assert u in train_has  # never test a user with no train ratings


def test_split_ratings_is_deterministic_and_seedable():
    users = np.arange(1, 6).repeat(4)
    items = np.tile(np.arange(50, 54), 5)
    ratings = np.ones(20)
    a = split_ratings(users, items, ratings, k=4, seed=7)
    b = split_ratings(users, items, ratings, k=4, seed=7)
    c = split_ratings(users, items, ratings, k=4, seed=8)
    for (ta, _), (tb, _) in zip(a, b):
        assert np.array_equal(ta[0], tb[0])
        assert np.array_equal(ta[1], tb[1])
    assert any(
        not np.array_equal(fold_a[0][0], fold_c[0][0]) for fold_a, fold_c in zip(a, c)
    )


def test_split_ratings_rejects_small_k():
    with pytest.raises(ValueError):
        split_ratings(np.array([1]), np.array([1]), np.array([1.0]), k=1)
