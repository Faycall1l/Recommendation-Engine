from recagent.agent import RankedItems, RecAgent, _tools, usage_summary
from recagent.config import LLMConfig


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


def test_tool_functions_are_registered():
    names = {fn.__name__ for fn in _tools()}
    assert names == {"recommend", "user_profile", "item_info", "search_items"}


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
