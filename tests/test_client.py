from types import SimpleNamespace

import scipy.sparse as sp

from recagent.agent import RankedItem, RankedItems
from recagent.cf import UserBasedCF
from recagent.client import RecClient, _to_recommendations
from recagent.config import LLMConfig
from recagent.explain import RecExplainer
from recagent.tools import ToolRegistry


def _user_cf_state():
    matrix = sp.csr_matrix([[5.0, 3.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0], [2.0, 0.0, 4.0, 0.0]])
    return {
        "model": UserBasedCF().fit(matrix),
        "matrix": matrix,
        "uid_to_idx": {1: 0, 2: 1, 3: 2},
        "iid_to_idx": {10: 0, 11: 1, 12: 2, 13: 3},
        "user_ids": [1, 2, 3],
        "item_ids": [10, 11, 12, 13],
        "items_meta": {
            10: {"title": "Alpha", "genres": ["Sci-Fi"]},
            11: {"title": "Beta", "genres": ["Drama"]},
            12: {"title": "Gamma", "genres": ["Comedy"]},
            13: {"title": "Delta", "genres": ["Sci-Fi"]},
        },
        "cf_kind": "user",
    }

DISABLED = LLMConfig(enabled=False)


class _StubModel:
    def recommend(self, matrix, user_idx, n=10):
        return [(0, 0.9), (1, 0.8), (2, 0.7)][:n]

    def similar_items(self, item_idx, n=10):
        return [(j, 0.5) for j in range(3) if j != item_idx][:n]

    def similar_users(self, user_idx, n=10):
        return [(j, 0.5) for j in range(1) if j != user_idx][:n]


def make_state():
    return {
        "model": _StubModel(),
        "matrix": sp.csr_matrix((1, 3)),
        "uid_to_idx": {1: 0},
        "iid_to_idx": {10: 0, 11: 1, 12: 2},
        "user_ids": [1],
        "item_ids": [10, 11, 12],
        "items_meta": {
            10: {"title": "Alpha", "genres": ["Sci-Fi"]},
            11: {"title": "Beta", "genres": ["Drama"]},
            12: {"title": "Gamma", "genres": ["Comedy"]},
        },
    }


def _fake_agent():
    class FakeUsage:
        requests = 1
        request_tokens = 10
        response_tokens = 5

    class FakeResult:
        output = RankedItems(items=[RankedItem(item_id=10, reason="fits profile")])

        def usage(self):
            return FakeUsage()

    return SimpleNamespace(run=lambda request, deps: FakeResult(), config=SimpleNamespace(model="x"))


def test_recommend_degrades_to_cf_without_agent(tmp_path):
    client = RecClient(
        state=make_state(),
        llm_config=DISABLED,
        feedback_path=tmp_path / "f.jsonl",
    )
    resp = client.recommend(1, k=2)
    assert [i.item_id for i in resp.items] == [10, 11]
    assert resp.items[0].reason == "collaborative filter"
    assert resp.usage == {}


def test_recommend_cold_start_uses_popularity_prior(tmp_path):
    client = RecClient(
        state=make_state(),
        llm_config=DISABLED,
        feedback_path=tmp_path / "f.jsonl",
    )
    resp = client.recommend(99999, k=2)
    assert resp.items
    assert resp.items[0].reason == "popularity prior (cold start)"


def test_recommend_with_fake_agent_maps_metadata(tmp_path):
    client = RecClient(state=make_state(), agent=_fake_agent(), feedback_path=tmp_path / "f.jsonl")
    resp = client.recommend(1, k=2)
    assert resp.items[0].item_id == 10
    assert resp.items[0].title == "Alpha"
    assert resp.items[0].genres == ["Sci-Fi"]
    assert resp.usage["requests"] == 1


def test_filters_request_builds_constraint():
    client = RecClient(state=make_state())
    assert "must be Sci-Fi" in client._filters_request(1, 5, {"genre": "Sci-Fi"})
    assert "must be" not in client._filters_request(1, 5, None)


def test_chat_with_fake_agent_builds_evidence(tmp_path):
    client = RecClient(state=make_state(), agent=_fake_agent(), feedback_path=tmp_path / "f.jsonl")
    resp = client.chat("sci-fi movies", user_id=1)
    assert resp.items[0].item_id == 10
    assert "User profile" in resp.evidence
    assert "Collaborative filtering" in resp.evidence


def test_chat_cold_start_user_builds_evidence(tmp_path):
    client = RecClient(state=make_state(), agent=_fake_agent(), feedback_path=tmp_path / "f.jsonl")
    resp = client.chat("sci-fi movies", user_id=99999)
    assert "Cold-start user" in resp.evidence
    assert "Popularity prior" in resp.evidence


def test_explain_and_health(tmp_path):
    client = RecClient(state=make_state(), feedback_path=tmp_path / "f.jsonl")
    info = client.explain(10)
    assert info["title"] == "Alpha"
    assert client.health()["n_users"] == 1
    assert client.health()["n_items"] == 3


class _FakeExplainer(RecExplainer):
    def __init__(self):
        pass

    def explain(self, explanation):
        return "grounded prose", {"requests": 1}

    async def aexplain(self, explanation):
        return "grounded prose", {"requests": 1}


def test_explain_recommendation_degrades_to_snippet(tmp_path):
    client = RecClient(state=make_state(), llm_config=DISABLED, feedback_path=tmp_path / "f.jsonl")
    resp = client.explain_recommendation(1, 11)
    assert resp.llm is False
    assert resp.text == resp.explanation.snippet
    assert resp.explanation.title == "Beta"


def test_explain_recommendation_uses_llm_when_available(tmp_path):
    client = RecClient(
        state=make_state(),
        explainer=_FakeExplainer(),
        feedback_path=tmp_path / "f.jsonl",
    )
    resp = client.explain_recommendation(1, 11)
    assert resp.llm is True
    assert resp.text == "grounded prose"
    assert resp.usage["requests"] == 1


def test_feedback_appends_jsonl(tmp_path):
    client = RecClient(state=make_state(), feedback_path=tmp_path / "f.jsonl")
    result = client.feedback(1, 10, liked=True)
    assert result["accepted"] is True
    lines = (tmp_path / "f.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert '"item_id":10' in lines[0]


def test_to_recommendations_maps_scores():
    deps = ToolRegistry(make_state())
    ranked = RankedItems(items=[RankedItem(item_id=10, reason="top")])
    items = _to_recommendations(ranked, deps, scores={10: 0.9})
    assert items[0].score == 0.9
    assert items[0].reason == "top"


def test_recommend_works_with_user_cf_engine(tmp_path):
    client = RecClient(
        state=_user_cf_state(),
        llm_config=DISABLED,
        feedback_path=tmp_path / "f.jsonl",
    )
    resp = client.recommend(1, k=2)
    assert [i.item_id for i in resp.items] == [12, 13]
    assert resp.items[0].title == "Gamma"
    assert resp.items[0].reason == "collaborative filter"


def test_tools_work_with_user_cf_engine():
    deps = ToolRegistry(_user_cf_state())
    assert [e.item_id for e in deps.recommend(1, n=2).items] == [12, 13]
    users = deps.similar_users(1, n=1).users
    assert [u.user_id for u in users] == [2]
    items = deps.similar_items(12, n=2).items
    assert [e.item_id for e in items] == [11]
    assert [e.item_id for e in deps.trending(2).items] == [12, 11]
