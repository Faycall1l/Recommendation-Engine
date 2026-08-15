import numpy as np
import pytest
import scipy.sparse as sp

from recagent.cf import CF_KINDS, ItemBasedCF, UserBasedCF, build_cf
from recagent.state import load_state, save_state


def _full_state(kind="user"):
    matrix = _matrix()
    model = build_cf(kind, matrix) if kind != "als" else None
    return {
        "model": model,
        "matrix": matrix,
        "uid_to_idx": {1: 0, 2: 1, 3: 2},
        "iid_to_idx": {11: 0, 12: 1, 13: 2, 14: 3},
        "user_ids": [1, 2, 3],
        "item_ids": [11, 12, 13, 14],
        "items_meta": {},
        "cf_kind": kind,
    }


def _matrix() -> sp.csr_matrix:
    return sp.csr_matrix(
        [
            [5.0, 3.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 0.0],
            [2.0, 0.0, 4.0, 0.0],
        ]
    )


def test_cf_kinds():
    assert set(CF_KINDS) == {"als", "user", "item"}


def test_userbased_fit_means_and_centered():
    cf = UserBasedCF().fit(_matrix())
    np.testing.assert_allclose(cf.user_means, [4.0, 2.0, 3.0])
    expected = np.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(cf.centered.toarray(), expected)
    assert cf.matrix.format == "csr"


def test_userbased_similarity_pearson_hand_case():
    cf = UserBasedCF(min_sim=0.0).fit(_matrix())
    sim = cf.similarity
    # u0'=[1,-1,0,0], u1'=[0,-1,1,0] -> dot=1, /sqrt(2)*sqrt(2) = 0.5
    assert sim[0, 1] == pytest.approx(0.5)
    # u0' vs u2'=[-1,0,1,0] -> dot=-1 -> -0.5, floored to 0 at min_sim=0
    assert sim[0, 2] == 0.0
    # u1' vs u2' -> dot=1 -> 0.5
    assert sim[1, 2] == pytest.approx(0.5)
    np.testing.assert_allclose(np.diag(sim), 0.0)


def test_userbased_similarity_identical_users():
    matrix = sp.csr_matrix(
        [
            [5.0, 3.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 0.0],
            [5.0, 3.0, 0.0, 0.0],  # identical to user 0
        ]
    )
    cf = UserBasedCF().fit(matrix)
    assert cf.similarity[0, 2] == pytest.approx(1.0)


def test_userbased_similarity_min_sim_floor():
    cf = UserBasedCF(min_sim=0.6).fit(_matrix())
    assert (cf.similarity == 0.0).all()


def test_userbased_predict_hand_case():
    cf = UserBasedCF(min_sim=0.0).fit(_matrix())
    # item 2: deviations [0, 1, 1]; sim[0]=[0, .5, 0] -> 4 + 0.5/0.5 = 5.0
    assert cf.predict(0, 2) == pytest.approx(5.0)
    # item 3 is unrated everywhere -> prediction collapses to the user mean
    assert cf.predict(0, 3) == pytest.approx(4.0)
    assert cf.predict(2, 3) == pytest.approx(3.0)


def test_userbased_predict_falls_back_to_mean_without_neighbours():
    matrix = sp.csr_matrix([[5.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    cf = UserBasedCF().fit(matrix)
    assert cf.predict(0, 1) == pytest.approx(5.0)


def test_userbased_recommend_hand_case():
    matrix = _matrix()
    cf = UserBasedCF(min_sim=0.0).fit(matrix)
    # user 0: scores [4,3,5,4], rated {0,1} -> unseen 2 (5.0), 3 (4.0)
    assert cf.recommend(matrix, 0, n=2) == [(2, 5.0), (3, 4.0)]
    # user 2: scores [3,2,4,3], rated {0,2} -> unseen 3 (3.0), 1 (2.0)
    assert cf.recommend(matrix, 2, n=2) == [(3, 3.0), (1, 2.0)]


def test_userbased_recommend_excludes_rated_and_caps_n():
    matrix = _matrix()
    cf = UserBasedCF(min_sim=0.0).fit(matrix)
    out = cf.recommend(matrix, 0, n=10)
    rated = {0, 1}
    ids = [idx for idx, _ in out]
    assert rated.isdisjoint(ids)
    assert len(out) == 2  # only two unseen items exist
    assert all(isinstance(score, float) for _, score in out)


def test_userbased_score_all_batch_matches_predict():
    matrix = _matrix()
    cf = UserBasedCF(min_sim=0.0).fit(matrix)
    all_scores = cf.score_all()
    assert all_scores.shape == matrix.shape
    np.testing.assert_allclose(all_scores[0], [4.0, 3.0, 5.0, 4.0])
    np.testing.assert_allclose(all_scores[2], [3.0, 2.0, 4.0, 3.0])
    for u in range(3):
        for j in range(4):
            assert all_scores[u, j] == pytest.approx(cf.predict(u, j))


def test_userbased_score_all_no_neighbours_is_user_mean():
    matrix = sp.csr_matrix([[5.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    cf = UserBasedCF().fit(matrix)
    all_scores = cf.score_all()
    np.testing.assert_allclose(all_scores[0], [5.0, 5.0, 5.0])
    assert np.isfinite(all_scores).all()


def test_userbased_roundtrip_save_load(tmp_path):
    matrix = _matrix()
    original = UserBasedCF(min_sim=0.1).fit(matrix)
    path = tmp_path / "model_cf.npz"
    original.save(path)

    restored = UserBasedCF.load(path)
    assert restored.min_sim == pytest.approx(0.1)
    np.testing.assert_allclose(restored.user_means, original.user_means)
    np.testing.assert_allclose(restored.similarity, original.similarity)
    np.testing.assert_allclose(restored.score_all(), original.score_all())
    np.testing.assert_allclose(restored.centered.toarray(), original.centered.toarray())
    assert restored.recommend(matrix, 0, n=2) == original.recommend(matrix, 0, n=2)


def test_itembased_fit_means_and_similarity():
    cf = ItemBasedCF(min_sim=0.0).fit(_matrix())
    # item means: (5+2)/2=3.5, (3+1)/2=2, (3+4)/2=3.5, unrated=0
    np.testing.assert_allclose(cf.item_means, [3.5, 2.0, 3.5, 0.0])
    sim = cf.similarity
    assert sim[0, 1] == pytest.approx(0.5)
    assert sim[0, 2] == 0.0  # -0.5 floored at min_sim=0
    assert sim[1, 2] == pytest.approx(0.5)
    np.testing.assert_allclose(np.diag(sim), 0.0)
    assert sim.shape == (4, 4)


def test_itembased_predict_hand_case():
    cf = ItemBasedCF(min_sim=0.0).fit(_matrix())
    # user 0 rates item1=3; item2 similar to item1 (0.5) -> (0.5*3)/0.5 = 3.0
    assert cf.predict(0, 2) == pytest.approx(3.0)
    # item3 is similar to nothing -> falls back to the user mean (4.0)
    assert cf.predict(0, 3) == pytest.approx(4.0)


def test_itembased_recommend_hand_case():
    matrix = _matrix()
    cf = ItemBasedCF(min_sim=0.0).fit(matrix)
    # user 0 rated {0:5, 1:3}; preds item0=3.0, item1=5.0, item2=3.0, item3=4.0
    # unseen: item2 (3.0), item3 (4.0) -> item3 first
    out = cf.recommend(matrix, 0, n=2)
    assert [idx for idx, _ in out] == [3, 2]
    assert out[0][1] == pytest.approx(4.0)
    assert out[1][1] == pytest.approx(3.0)


def test_itembased_score_all_matches_predict():
    cf = ItemBasedCF(min_sim=0.0).fit(_matrix())
    all_scores = cf.score_all()
    assert all_scores.shape == cf.matrix.shape
    assert all_scores[0, 2] == pytest.approx(cf.predict(0, 2))
    assert all_scores[0, 3] == pytest.approx(cf.predict(0, 3))
    assert all_scores[2, 0] == pytest.approx(cf.predict(2, 0))
    assert np.isfinite(all_scores).all()


def test_itembased_roundtrip_save_load(tmp_path):
    matrix = _matrix()
    original = ItemBasedCF(min_sim=0.1).fit(matrix)
    path = tmp_path / "model_cf.npz"
    original.save(path)

    restored = ItemBasedCF.load(path)
    assert restored.min_sim == pytest.approx(0.1)
    np.testing.assert_allclose(restored.similarity, original.similarity)
    np.testing.assert_allclose(restored.user_means, original.user_means)
    np.testing.assert_allclose(restored.item_means, original.item_means)
    assert restored.predict(0, 2) == pytest.approx(original.predict(0, 2))
    assert restored.recommend(matrix, 0, n=2) == original.recommend(matrix, 0, n=2)
    np.testing.assert_allclose(restored.score_all(), original.score_all())


def test_legacy_dense_similarity_load(tmp_path):
    matrix = _matrix()
    for cls in (UserBasedCF, ItemBasedCF):
        model = cls(min_sim=0.1).fit(matrix)
        # write the legacy format: dense similarity, no sim_sparse keys
        coo = model.matrix.tocoo()
        payload: dict = {
            "min_sim": model.min_sim,
            "m_data": coo.data,
            "m_row": coo.row,
            "m_col": coo.col,
            "m_shape": np.asarray(coo.shape),
            "similarity": np.asarray(model.similarity),
        }
        if cls is UserBasedCF:
            payload["user_means"] = model.user_means
            payload.update(
                c_data=model.centered.tocoo().data,
                c_row=model.centered.tocoo().row,
                c_col=model.centered.tocoo().col,
                c_shape=np.asarray(model.centered.shape),
            )
        else:
            payload["user_means"] = model.user_means
            payload["item_means"] = model.item_means
        path = tmp_path / f"legacy_{cls.__name__}.npz"
        np.savez(path, **payload)

        restored = cls.load(path)
        np.testing.assert_allclose(restored.similarity, model.similarity)
        assert restored.recommend(matrix, 0, n=2) == model.recommend(matrix, 0, n=2)


def test_build_cf_factory():
    assert isinstance(build_cf("user", _matrix()), UserBasedCF)
    assert isinstance(build_cf("item", _matrix()), ItemBasedCF)
    assert isinstance(build_cf("USER", _matrix()), UserBasedCF)
    with pytest.raises(ValueError):
        build_cf("als", _matrix())
    with pytest.raises(ValueError):
        build_cf("bogus", _matrix())


def test_train_from_data_rejects_invalid_cf():
    from recagent.model import train_from_data

    with pytest.raises(ValueError, match="cf must be one of"):
        train_from_data(cf="bogus")


@pytest.mark.parametrize("kind", ["user", "item"])
def test_state_roundtrip_engine_agnostic(tmp_path, kind):
    state = _full_state(kind)
    save_state(state, tmp_path)
    restored = load_state(tmp_path)
    assert restored["cf_kind"] == kind
    assert type(restored["model"]).__name__ == ("UserBasedCF" if kind == "user" else "ItemBasedCF")
    np.testing.assert_allclose(restored["matrix"].toarray(), state["matrix"].toarray())
    out = restored["model"].recommend(state["matrix"], 0, n=2)
    expected = [3, 2] if kind == "item" else [2, 3]
    assert [idx for idx, _ in out] == expected


def test_userbased_sparse_topk_matches_dense_top_entries():
    dense = UserBasedCF(min_sim=0.0).fit(_matrix())
    sparse = UserBasedCF(min_sim=0.0, topk=1).fit(_matrix())
    assert sp.issparse(sparse.similarity)
    # sparse kNN keeps the single strongest positive entry per row
    row = sparse._row(0)
    assert (row != 0.0).sum() == 1
    assert max(row) == pytest.approx(max(dense.similarity[0]))


def test_userbased_sparse_topk_recommend_and_roundtrip(tmp_path):
    matrix = _matrix()
    original = UserBasedCF(min_sim=0.0, topk=1).fit(matrix)
    assert original.recommend(matrix, 0, n=2) == [(2, 5.0), (3, 4.0)]
    assert original.predict(0, 2) == pytest.approx(5.0)
    path = tmp_path / "sparse_user.npz"
    original.save(path)
    restored = UserBasedCF.load(path)
    assert restored.topk == 1
    assert sp.issparse(restored.similarity)
    assert restored.recommend(matrix, 0, n=2) == original.recommend(matrix, 0, n=2)
    np.testing.assert_allclose(restored._row(0), original._row(0))


def test_itembased_sparse_topk_recommend_and_roundtrip(tmp_path):
    matrix = _matrix()
    original = ItemBasedCF(min_sim=0.0, topk=2).fit(matrix)
    assert sp.issparse(original.similarity)
    assert original.recommend(matrix, 0, n=2)[0][0] == 3
    path = tmp_path / "sparse_item.npz"
    original.save(path)
    restored = ItemBasedCF.load(path)
    assert restored.topk == 2
    assert sp.issparse(restored.similarity)
    assert restored.recommend(matrix, 0, n=2) == original.recommend(matrix, 0, n=2)


def test_build_cf_topk_passthrough():
    cf = build_cf("item", _matrix(), topk=2)
    assert sp.issparse(cf.similarity)
    assert max(cf.similarity.getnnz(axis=1)) <= 2
