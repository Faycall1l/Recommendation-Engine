import typing

from recagent.agent import (
    RankedItem,
    RankedItems,
    RecAgent,
    _clean_items,
    build_evidence,
    build_plan,
    detect_genre,
    usage_summary,
)
from recagent.config import LLMConfig
from recagent.tools import ToolRegistry
from tests.test_tools import build_state


class _FakeDeps:
    items_meta: typing.ClassVar[dict] = {
        1: {"genres": ["Sci-Fi", "Drama"]},
        2: {"genres": ["Comedy"]},
    }


def test_ranked_items_validation():
    parsed = RankedItems.model_validate(
        {"items": [{"item_id": 7, "reason": "top CF score"}, {"item_id": 3, "reason": "genre match"}]}
    )
    assert parsed.items[0].item_id == 7
    assert parsed.items[1].reason == "genre match"


def test_agent_constructs_without_network():
    config = LLMConfig(enabled=True, base_url="http://localhost:9/v1", api_key="test-key", model="x")
    agent = RecAgent(config, state={})
    assert agent.agent.name == "recagent"


def test_detect_genre_aliases():
    deps = _FakeDeps()
    assert detect_genre("every item must be sci-fi", deps) == "Sci-Fi"
    assert detect_genre("science fiction please", deps) == "Sci-Fi"
    assert detect_genre("comedy only", deps) == "Comedy"
    assert detect_genre("something unrelated", deps) is None


def test_build_plan_parses_request():
    deps = _FakeDeps()
    plan = build_plan("Recommend 3 items for user_id: 42. sci fi only", deps)
    assert plan == {"user_id": 42, "k": 3, "genre": "Sci-Fi"}
    plan = build_plan("Recommend 5 items for user_id: 7.", deps)
    assert plan == {"user_id": 7, "k": 5, "genre": None}


def test_clean_items_drops_hallucinations_violations_and_duplicates():
    plan = {"user_id": 1, "k": 5, "genre": "Sci-Fi"}
    meta = {1: {"Sci-Fi"}, 2: {"Drama"}, 3: {"Sci-Fi"}}
    items = [
        RankedItem(item_id=1, reason="a"),
        RankedItem(item_id=99, reason="hallucinated"),
        RankedItem(item_id=2, reason="wrong genre"),
        RankedItem(item_id=1, reason="duplicate"),
        RankedItem(item_id=3, reason="ok"),
    ]
    kept = _clean_items(items, plan=plan, meta=meta)
    assert [it.item_id for it in kept] == [1, 3]


def test_usage_summary_empty():
    class FakeUsage:
        def __init__(self):
            self.requests = 0
            self.request_tokens = 0
            self.response_tokens = 0

    class FakeResult:
        def usage(self):
            return FakeUsage()

    assert usage_summary(FakeResult()) == {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}


def test_evidence_warm_user():
    deps = ToolRegistry(build_state())
    text, meta = build_evidence({"user_id": 101, "k": 5, "genre": None}, deps)
    assert "User profile" in text
    assert "Collaborative filtering candidates" in text
    assert len(meta) > 0


def test_evidence_cold_start_user():
    deps = ToolRegistry(build_state())
    text, _ = build_evidence({"user_id": 99999, "k": 5, "genre": "Action"}, deps)
    assert "Cold-start user" in text
    assert "Popularity prior" in text
    assert "Search matches" in text


def test_evidence_rare_genre_widens_via_similar_items():
    deps = ToolRegistry(build_state())
    text, meta = build_evidence({"user_id": 101, "k": 12, "genre": "Action"}, deps)
    assert "Widened candidates" in text
    assert len(meta) >= 12
