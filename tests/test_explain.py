import numpy as np
import pytest
import scipy.sparse as sp

from recagent.cf import UserBasedCF
from recagent.explain import (
    _COLD_START,
    explain_recommendation,
    user_top_genres,
)
from recagent.state import load_state, save_state
from recagent.tools import ToolRegistry


@pytest.fixture()
def deps(tmp_path):
    matrix = sp.csr_matrix(
        np.asarray(
            [
                # u1: sci-fi fan; Dune unrated, Star Wars/Alien rated 5
                [5, 0, 1, 1, 5, 0, 0],
                # u2: shares sci-fi taste
                [4, 4, 0, 2, 4, 1, 0],
                # u3: sci-fi, weaker
                [5, 3, 0, 0, 2, 0, 0],
                # u4: no sci-fi at all (drama/thriller/noir/kids)
                [0, 0, 5, 5, 0, 4, 3],
            ],
            dtype=float,
        )
    )
    state = {
        "model": UserBasedCF().fit(matrix),
        "matrix": matrix,
        "uid_to_idx": {1: 0, 2: 1, 3: 2, 4: 3},
        "iid_to_idx": {100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 105: 5, 106: 6},
        "user_ids": [1, 2, 3, 4],
        "item_ids": [100, 101, 102, 103, 104, 105, 106],
        "items_meta": {
            100: {"title": "Star Wars", "genres": ["Sci-Fi", "Action"]},
            101: {"title": "Dune", "genres": ["Sci-Fi"]},
            102: {"title": "Casablanca", "genres": ["Drama"]},
            103: {"title": "Jaws", "genres": ["Thriller"]},
            104: {"title": "Alien", "genres": ["Sci-Fi", "Horror"]},
            105: {"title": "Maltese Falcon", "genres": ["Film-Noir"]},
            106: {"title": "Dumbo", "genres": ["Children"]},
        },
        "cf_kind": "user",
    }
    save_state(state, tmp_path)
    return ToolRegistry(load_state(str(tmp_path)))


def test_user_top_genres_weighted(deps):
    genres = user_top_genres(deps, 1)
    assert genres[0] == "Sci-Fi"  # 10 points beats Drama's 2
    assert set(genres) >= {"Sci-Fi", "Action", "Drama"}


def test_explain_genre_affinity(deps):
    # Dune is unseen by user 1 and in their dominant genre
    expl = explain_recommendation(deps, 1, 101)
    assert expl.basis == "genre-affinity"
    assert expl.matched_genres == ["Sci-Fi"]
    liked_ids = {e.item_id for e in expl.user_likes}
    assert {100, 104} <= liked_ids  # Star Wars, Alien — both rated 5
    rated5 = {e.item_id for e in expl.user_likes if e.rating == 5.0}
    assert {100, 104} <= rated5
    assert expl.user_mean == pytest.approx(3.0)
    assert expl.score is not None and expl.score > expl.user_mean
    assert expl.boost == pytest.approx(expl.score - expl.user_mean)
    assert expl.title == "Dune"
    # the snippet only repeats computed facts
    assert "Dune" in expl.snippet


def test_explain_similar_evidence_mechanism(deps):
    # user 1 rated Alien, which is item-item similar to Dune (co-rated signal)
    expl = explain_recommendation(deps, 1, 101)
    assert 104 in {e.item_id for e in expl.similar_rated}


def test_explain_taste_overlap(deps):
    # Dune vs user 4's non-sci-fi profile: no genre overlap, nothing similar
    # rated -> falls back to generic taste overlap
    expl = explain_recommendation(deps, 4, 101)
    assert expl.basis == "taste-overlap"
    assert expl.matched_genres == []
    assert expl.similar_rated == []
    assert expl.user_likes  # their profile items are still cited
    assert "Dune" in expl.snippet


def test_explain_cold_start(deps):
    expl = explain_recommendation(deps, 999, 101)
    assert expl.basis == _COLD_START
    assert expl.user_mean is None
    assert expl.matched_genres == []
    assert "Dune" in expl.snippet
