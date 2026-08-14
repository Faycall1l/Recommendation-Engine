"""The agentic recommender: a pydantic-ai Agent over a vLLM-served Gemma-4.

The agent plans, calls collaborative-filtering tools for evidence, and emits a
typed, structured ranked list — so the same agent that "chats" can be scored
offline against the raw CF baseline.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from recagent.config import LLMConfig
from recagent.tools import ItemEntry, ItemList, ToolRegistry


class RankedItem(BaseModel):
    item_id: int = Field(description="item id from the catalog")
    reason: str = Field(description="one-sentence justification grounded in evidence")


class RankedItems(BaseModel):
    items: list[RankedItem] = Field(description="ranked list, best first")


SYSTEM_PROMPT = """\
You are RecAgent, a recommender agent. You rank items for a user by grounding
every choice in evidence from the collaborative filtering engine.

Workflow — call the tools, do not skip them:
1. user_profile(user_id) to learn the user's taste.
2. recommend(user_id, n) to get candidates ranked by collaborative signal.
3. item_info / search_items when you need more detail or to satisfy a
   natural-language constraint (genre, era, mood) from the request.
4. Decide the final ranking yourself: the engine is strong on signal but blind
   to semantics — you reconcile it with stated taste and constraints.

Rules:
- Never output an item the user has already seen (recommend already filters these).
- Never invent item ids or titles: output only item_ids you actually observed.
- Output the full requested number of items, best first.
- A reason is evidence + judgement, e.g. "top CF score and matches your sci-fi
  streak" — never a bare echo of the title.
"""


def _tools() -> list[Any]:
    async def recommend(
        ctx: RunContext[ToolRegistry], user_id: int, n: int = 10
    ) -> ItemList:
        """Top-n candidate items for a user from the CF engine, best first."""
        return ctx.deps.recommend(user_id, n)

    async def user_profile(
        ctx: RunContext[ToolRegistry], user_id: int, k: int = 8
    ) -> ItemList:
        """The user's highest-rated items — their taste profile."""
        return ctx.deps.user_profile(user_id, k)

    async def item_info(ctx: RunContext[ToolRegistry], item_id: int) -> ItemEntry:
        """Metadata for a single item: title, genres, popularity."""
        return ctx.deps.item_info(item_id)

    async def search_items(
        ctx: RunContext[ToolRegistry], query: str, n: int = 10
    ) -> ItemList:
        """Items whose title or genres match a free-text query."""
        return ctx.deps.search_items(query, n)

    return [recommend, user_profile, item_info, search_items]


class RecAgent:
    """Typed tool-using recommender agent backed by an OpenAI-compatible vLLM model."""

    def __init__(
        self,
        config: LLMConfig,
        state: dict[str, Any],
        *,
        temperature: float = 0.2,
        max_steps: int = 12,
    ):
        self.config = config
        self.state = state
        model = OpenAIChatModel(
            config.model,
            provider=OpenAIProvider(base_url=config.base_url, api_key=config.api_key),
        )
        self.agent = Agent(
            model,
            name="recagent",
            deps_type=ToolRegistry,
            output_type=RankedItems,
            system_prompt=SYSTEM_PROMPT,
            tools=_tools(),
            retries=2,
            defer_model_check=True,
            model_settings=ModelSettings(
                temperature=temperature, max_tokens=1500, parallel_tool_calls=True
            ),
        )

    async def arun(self, request: str, deps: ToolRegistry) -> Any:
        """Async single-task run; returns a pydantic-ai AgentRunResult."""
        return await self.agent.run(request, deps=deps)

    def run(self, request: str, deps: ToolRegistry) -> Any:
        """Sync convenience wrapper for the CLI."""
        return asyncio.run(self.arun(request, deps))


def usage_summary(result: Any) -> dict:
    usage = result.usage()
    return {
        "requests": usage.requests,
        "prompt_tokens": usage.request_tokens,
        "completion_tokens": usage.response_tokens,
    }
