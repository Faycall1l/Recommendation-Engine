import typing

import pytest

from recagent.agent import (
    RankedItem,
    RankedItems,
    ReasoningTrace,
    RecAgent,
    _clean_items,
    _jaccard,
    _reflect_on_ranking,
    build_evidence,
    build_plan,
    detect_genre,
    mmr_rerank,
)
from recagent.config import LLMConfig, RecAgentConfig
from recagent.tools import ToolRegistry
from recagent.utils import usage_summary
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


def test_agent_reflect_flag():
    config = LLMConfig(enabled=True, base_url="http://localhost:9/v1", api_key="test-key", model="x")
    agent_on = RecAgent(config, state={}, reflect=True)
    agent_off = RecAgent(config, state={}, reflect=False)
    assert agent_on.reflect is True
    assert agent_off.reflect is False


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
    text, meta, sections = build_evidence({"user_id": 101, "k": 5, "genre": None}, deps)
    assert "User rating context" in text
    assert "mean rating" in text
    assert "User profile" in text
    assert "Collaborative filtering candidates" in text
    assert "Social proof" in text
    assert "watched " in text
    assert len(meta) > 0
    assert set(sections) == {"profile", "candidates", "social"}


def test_evidence_cold_start_user():
    deps = ToolRegistry(build_state())
    text, _, sections = build_evidence({"user_id": 99999, "k": 5, "genre": "Action"}, deps)
    assert "Cold-start user" in text
    assert "Popularity prior" in text
    assert "Search matches" in text
    assert "trending" in sections


def test_evidence_rare_genre_widens_via_similar_items():
    deps = ToolRegistry(build_state())
    text, meta, sections = build_evidence({"user_id": 101, "k": 12, "genre": "Action"}, deps)
    assert "Widened candidates" in text
    assert len(meta) >= 12
    assert "genre_search" in sections


# ---------- reflection tests ----------

def test_reflect_no_issues_on_good_ranking():
    plan = {"user_id": 1, "k": 3, "genre": None}
    meta = {1: {"Drama"}, 2: {"Comedy"}, 3: {"Action"}}
    sections = {"candidates": [1, 2, 3]}
    items = [
        RankedItem(item_id=1, reason="best"),
        RankedItem(item_id=2, reason="good"),
        RankedItem(item_id=3, reason="ok"),
    ]
    report = _reflect_on_ranking(items, plan=plan, meta=meta, evidence_sections=sections)
    assert report.needs_refinement is False
    assert report.issues == []


def test_reflect_flags_too_few_after_cleaning():
    plan = {"user_id": 1, "k": 5, "genre": "Sci-Fi"}
    meta = {1: {"Sci-Fi"}, 2: {"Sci-Fi"}}  # only 2 valid items
    sections = {"candidates": [1, 2]}
    items = [
        RankedItem(item_id=1, reason="a"),
        RankedItem(item_id=2, reason="b"),
    ]
    report = _reflect_on_ranking(items, plan=plan, meta=meta, evidence_sections=sections)
    assert report.needs_refinement is True
    assert any("Only 2 of 5" in i for i in report.issues)


def test_reflect_flags_constraint_violations_in_raw():
    plan = {"user_id": 1, "k": 3, "genre": "Sci-Fi"}
    meta = {1: {"Sci-Fi"}, 2: {"Drama"}, 3: {"Sci-Fi"}}
    sections = {"candidates": [1, 2, 3]}
    items = [
        RankedItem(item_id=1, reason="ok"),
        RankedItem(item_id=2, reason="violation"),
        RankedItem(item_id=3, reason="ok"),
    ]
    report = _reflect_on_ranking(items, plan=plan, meta=meta, evidence_sections=sections)
    assert report.needs_refinement is True
    assert any("violate" in i and "Sci-Fi" in i for i in report.issues)


def test_reflect_flags_missing_evidence_section():
    plan = {"user_id": 1, "k": 2, "genre": None}
    meta = {1: {"Drama"}, 2: {"Comedy"}, 10: {"Action"}}
    sections = {"candidates": [1, 2], "social": [10]}
    # pick only from candidates, ignore social entirely
    items = [
        RankedItem(item_id=1, reason="a"),
        RankedItem(item_id=2, reason="b"),
    ]
    report = _reflect_on_ranking(items, plan=plan, meta=meta, evidence_sections=sections)
    assert report.needs_refinement is True
    assert any("social" in i for i in report.issues)


def test_reflect_flags_low_diversity():
    plan = {"user_id": 1, "k": 3, "genre": None}
    meta = {1: {"Sci-Fi"}, 2: {"Sci-Fi"}, 3: {"Sci-Fi"}}
    sections = {"candidates": [1, 2, 3]}
    items = [
        RankedItem(item_id=1, reason="a"),
        RankedItem(item_id=2, reason="b"),
        RankedItem(item_id=3, reason="c"),
    ]
    report = _reflect_on_ranking(items, plan=plan, meta=meta, evidence_sections=sections)
    assert report.needs_refinement is True
    assert any("identical genre" in i for i in report.issues)


def test_reflect_single_item_no_diversity_flag():
    plan = {"user_id": 1, "k": 1, "genre": None}
    meta = {1: {"Sci-Fi"}}
    sections = {"candidates": [1]}
    items = [RankedItem(item_id=1, reason="only one")]
    report = _reflect_on_ranking(items, plan=plan, meta=meta, evidence_sections=sections)
    assert report.needs_refinement is False


# ---------- MMR diversity tests ----------


def test_jaccard_basic():
    assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert _jaccard({"a"}, {"b"}) == 0.0
    assert _jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert _jaccard(set(), set()) == 0.0


def test_mmr_preserves_single_item():
    items = [RankedItem(item_id=1, reason="solo")]
    assert mmr_rerank(items, {1: {"Drama"}}) == items


def test_mmr_preserves_order_with_identical_genres():
    items = [
        RankedItem(item_id=1, reason="a"),
        RankedItem(item_id=2, reason="b"),
        RankedItem(item_id=3, reason="c"),
    ]
    meta = {1: {"Sci-Fi"}, 2: {"Sci-Fi"}, 3: {"Sci-Fi"}}
    reranked = mmr_rerank(items, meta)
    # all identical genres -> max_sim always 1.0 -> score = λ*relevance - (1-λ)*1.0
    # which is monotonically decreasing with relevance, so order preserved
    assert [it.item_id for it in reranked] == [1, 2, 3]


def test_mmr_promotes_diverse_items():
    # item 1: Sci-Fi, item 2: Sci-Fi (duplicate genre), item 3: Comedy (diverse)
    items = [
        RankedItem(item_id=1, reason="a"),
        RankedItem(item_id=2, reason="b"),
        RankedItem(item_id=3, reason="c"),
    ]
    meta = {1: {"Sci-Fi"}, 2: {"Sci-Fi"}, 3: {"Comedy"}}
    reranked = mmr_rerank(items, meta, lambda_param=0.5)
    ids = [it.item_id for it in reranked]
    # item 3 (Comedy) should be selected before item 2 (Sci-Fi) for diversity
    assert ids.index(3) < ids.index(2)


def test_mmr_lambda_zero_is_pure_diversity():
    items = [
        RankedItem(item_id=1, reason="a"),
        RankedItem(item_id=2, reason="b"),
        RankedItem(item_id=3, reason="c"),
    ]
    meta = {1: {"Sci-Fi"}, 2: {"Sci-Fi"}, 3: {"Comedy"}}
    reranked = mmr_rerank(items, meta, lambda_param=0.0)
    ids = [it.item_id for it in reranked]
    # with λ=0, diversity dominates: first item always selected,
    # then the most dissimilar item is picked next
    assert ids[0] == 1  # always first
    assert ids[1] == 3  # Comedy is most dissimilar to Sci-Fi


def test_mmr_lambda_one_is_pure_relevance():
    items = [
        RankedItem(item_id=1, reason="a"),
        RankedItem(item_id=2, reason="b"),
        RankedItem(item_id=3, reason="c"),
    ]
    meta = {1: {"Sci-Fi"}, 2: {"Comedy"}, 3: {"Drama"}}
    reranked = mmr_rerank(items, meta, lambda_param=1.0)
    ids = [it.item_id for it in reranked]
    # with λ=1, pure relevance: order unchanged regardless of genre diversity
    assert ids == [1, 2, 3]


def test_clean_items_diversity_false_skips_mmr():
    plan = {"user_id": 1, "k": 5, "genre": None}
    meta = {1: {"Sci-Fi"}, 2: {"Sci-Fi"}, 3: {"Comedy"}}
    items = [
        RankedItem(item_id=1, reason="a"),
        RankedItem(item_id=2, reason="b"),
        RankedItem(item_id=3, reason="c"),
    ]
    kept = _clean_items(items, plan=plan, meta=meta, diversity=False)
    assert [it.item_id for it in kept] == [1, 2, 3]


def test_clean_items_diversity_reorders():
    plan = {"user_id": 1, "k": 5, "genre": None}
    meta = {1: {"Sci-Fi"}, 2: {"Sci-Fi"}, 3: {"Comedy"}}
    items = [
        RankedItem(item_id=1, reason="a"),
        RankedItem(item_id=2, reason="b"),
        RankedItem(item_id=3, reason="c"),
    ]
    kept = _clean_items(items, plan=plan, meta=meta, diversity=True)
    ids = [it.item_id for it in kept]
    # diverse item (Comedy) should be promoted ahead of duplicate genre
    assert ids.index(3) < ids.index(2)


# ── ReasoningTrace tests ────────────────────────────────────────────


def test_reasoning_trace_defaults():
    t = ReasoningTrace()
    assert t.request_id == ""
    assert t.plan_user_id is None
    assert t.raw_llm_output == ""
    assert t.reflection_issues == []
    assert t.refinement_applied is False
    assert t.latency_ms == 0.0


def test_reasoning_trace_with_values():
    t = ReasoningTrace(
        request_id="req-123",
        plan_user_id=42,
        plan_k=5,
        cleaned_item_ids=[1, 2, 3],
        latency_ms=150.5,
    )
    assert t.request_id == "req-123"
    assert t.plan_user_id == 42
    assert t.cleaned_item_ids == [1, 2, 3]


# ── RecAgentConfig tests ────────────────────────────────────────────


def test_rec_agent_config_defaults():
    cfg = RecAgentConfig()
    assert cfg.temperature == 0.1
    assert cfg.max_requests == 12
    assert cfg.reflect is True
    assert cfg.diversity is True
    assert cfg.lambda_param == 0.5
    assert cfg.evidence_budget_tokens == 4000


def test_rec_agent_config_custom():
    cfg = RecAgentConfig(temperature=0.5, max_requests=20, reflect=False)
    assert cfg.temperature == 0.5
    assert cfg.max_requests == 20
    assert cfg.reflect is False


def test_llm_config_requires_api_key_when_enabled():
    with pytest.raises(ValueError, match="api_key"):
        LLMConfig(enabled=True, base_url="http://localhost:9/v1", api_key="", model="x")


def test_llm_config_allows_empty_api_key_when_disabled():
    cfg = LLMConfig(enabled=False, base_url="http://localhost:9/v1", api_key="", model="x")
    assert cfg.enabled is False


def test_build_evidence_budget_truncates():
    deps = ToolRegistry(build_state())
    text, _, _ = build_evidence(
        {"user_id": 101, "k": 5, "genre": None}, deps, budget_tokens=50
    )
    assert len(text) <= 50 * 4 + 100  # some slack for the truncation marker
    assert "[...evidence truncated" in text
