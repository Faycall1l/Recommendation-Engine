"""The agentic recommender: a plan-then-execute agent over a vLLM-served Gemma-4.

Plan   — the harness parses the request (user, item count, any genre constraint).
Execute— the harness gathers evidence through the CF tool registry: the user's
         taste profile, engine-ranked candidates, and constraint-matched items.
Reason — one structured call to the LLM produces a typed, grounded ranked list.

Tools stay first-class (they are planned and executed); running them from a
deterministic harness instead of in the model's loop is dramatically more
reliable against the vLLM tool-calling template, and it is still scored on the
exact lists the agent emits.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from recagent.config import LLMConfig
from recagent.tools import ToolRegistry


class RankedItem(BaseModel):
    item_id: int = Field(description="item id from the catalog")
    reason: str = Field(description="one-sentence justification grounded in evidence")


class RankedItems(BaseModel):
    items: list[RankedItem] = Field(description="ranked list, best first")


SYSTEM_PROMPT = """\
You are RecAgent, a recommender agent. You produce ranked item lists grounded in
an evidence block: a user profile, collaborative-filtering candidates, and
optional search matches for an explicit constraint.

Rules:
- Choose items only from the evidence block. Never invent item ids.
- Honor explicit constraints. If the request says every item must be a genre,
  every output item must carry that genre.
- Rank best first: blend the engine score with fit to the user's taste and the
  constraint. Prefer items with strong popularity and average rating when scores
  are close; items liked by the user's most similar neighbours are strong fits.
- Justify each pick in one short sentence using the evidence.
- Do not repeat an item. Output as many items as requested, but fewer is better
  than padding with items that violate a constraint or are not in the evidence.
"""


def _genre_aliases(genre: str) -> list[str]:
    """Lowercased search forms for a catalog genre label."""
    base = genre.lower()
    if base == "sci-fi":
        return ["sci-fi", "sci fi", "scifi", "science fiction"]
    if base == "film-noir":
        return ["film-noir", "film noir", "noir"]
    if base == "children's":
        return ["children's", "children", "kids"]
    return [base]


def detect_genre(request: str, deps: ToolRegistry) -> str | None:
    """The catalog genre a free-text request refers to, if any."""
    haystack = request.lower()
    genres = {g for info in deps.items_meta.values() for g in info.get("genres", [])}
    for genre in sorted(genres):
        if any(alias in haystack for alias in _genre_aliases(genre)):
            return genre
    return None


def build_plan(request: str, deps: ToolRegistry, default_k: int = 5) -> dict[str, Any]:
    """Parse the request into a deterministic execution plan."""
    user_match = re.search(r"user_id:\s*(\d+)", request)
    if not user_match:
        raise ValueError(f"no user_id in request: {request!r}")
    k_match = re.search(r"(\d+)\s*items", request)
    return {
        "user_id": int(user_match.group(1)),
        "k": int(k_match.group(1)) if k_match else default_k,
        "genre": detect_genre(request, deps),
    }


def _fmt(entry: Any, detail: str | None = None) -> str:
    genres = ", ".join(entry.genres) or "-"
    parts = [f"- {entry.item_id}: {entry.title} [{genres}]"]
    if entry.rating_count:
        parts.append(f"watched {entry.rating_count}x")
    if entry.avg_rating is not None:
        parts.append(f"avg {entry.avg_rating:.1f}")
    if detail:
        parts.append(detail)
    return ", ".join(parts)


def build_evidence(plan: dict[str, Any], deps: ToolRegistry) -> tuple[str, dict[int, set[str]]]:
    """Gather evidence through the tools; return (text block, item genre map).

    Adaptive retrieval: warm users get profile + CF candidates + social proof;
    cold users fall back to a popularity prior; a rare genre widens the pool
    through item-item neighbours of the genre hits. Every item carries its
    popularity (times watched) and average rating so the model can weigh quality
    against raw score.
    """
    meta: dict[int, set[str]] = {}
    uid, k = plan["user_id"], plan["k"]
    genre = plan.get("genre")

    def absorb(items: list[Any]) -> None:
        for entry in items:
            meta[entry.item_id] = set(entry.genres)

    lines: list[str] = []
    if uid in deps.uid_to_idx:
        stats = deps.user_stats(uid)
        lines.append(
            f"User rating context (user_id: {uid}): {stats.n_rated} movies rated, "
            f"mean rating {stats.mean_rating:.2f} (1-5 scale)."
        )
        profile = deps.user_profile(uid, k=min(8, max(4, k)))
        lines.append("User profile (highest rated):")
        for entry in profile.items:
            lines.append(_fmt(entry, detail=f"rating {entry.rating}"))
        absorb(profile.items)

        candidates = deps.recommend(uid, n=20)
        lines.append(
            "Collaborative filtering candidates (engine score, best first; "
            "scores are comparable within this list, higher is stronger):"
        )
        for entry in candidates.items:
            lines.append(_fmt(entry, detail=f"score {entry.score}"))
        absorb(candidates.items)

        social = deps.similar_users(uid, n=3)
        if social.users:
            lines.append("Social proof (what the most similar users like):")
            for peer in social.users:
                peer_profile = deps.user_profile(peer.user_id, k=3)
                for entry in peer_profile.items:
                    lines.append(
                        _fmt(entry, detail=f"liked by similar user {peer.user_id}")
                    )
                absorb(peer_profile.items)
    else:
        lines.append(f"Cold-start user (user_id: {uid}) — no interaction history.")
        prior = deps.trending(n=20)
        lines.append("Popularity prior (most-watched across all users):")
        for entry in prior.items:
            lines.append(_fmt(entry, detail=f"rated {entry.rating_count}x"))
        absorb(prior.items)

    if genre:
        hits = deps.search_items(genre, n=15)
        lines.append(f"Search matches for the requested genre ({genre}):")
        for entry in hits.items:
            lines.append(_fmt(entry))
        absorb(hits.items)

        if len(hits.items) < k:
            widened: dict[int, Any] = {}
            for entry in hits.items:
                for neighbour in deps.similar_items(entry.item_id, n=3).items:
                    widened.setdefault(neighbour.item_id, neighbour)
            if widened:
                lines.append("Widened candidates (similar to the genre hits):")
                for entry in list(widened.values())[:15]:
                    lines.append(_fmt(entry, detail=f"similarity {entry.score}"))
                absorb(list(widened.values()))

    return "\n".join(lines), meta


def _clean_items(
    items: list[RankedItem],
    *,
    plan: dict[str, Any],
    meta: dict[int, set[str]],
) -> list[RankedItem]:
    """Drop hallucinations, constraint violations, and duplicates; cap at k."""
    seen: set[int] = set()
    kept: list[RankedItem] = []
    genre = plan.get("genre")
    for item in items:
        genres = meta.get(item.item_id)
        if genres is None or item.item_id in seen:
            continue
        if genre and genre not in genres:
            continue
        seen.add(item.item_id)
        kept.append(item)
    return kept[: plan["k"]]


class RecAgent:
    """Plan-then-execute recommender agent backed by an OpenAI-compatible vLLM model."""

    def __init__(
        self,
        config: LLMConfig,
        state: dict[str, Any],
        *,
        temperature: float = 0.1,
        max_requests: int = 12,
    ):
        self.config = config
        self.state = state
        self.max_requests = max_requests
        model = OpenAIChatModel(
            config.model,
            provider=OpenAIProvider(base_url=config.base_url, api_key=config.api_key),
        )
        self.agent = Agent(
            model,
            name="recagent",
            output_type=RankedItems,
            system_prompt=SYSTEM_PROMPT,
            retries=2,
            defer_model_check=True,
            model_settings=ModelSettings(
                temperature=temperature, max_tokens=2000, parallel_tool_calls=False
            ),
        )

    async def arun(self, request: str, deps: ToolRegistry) -> Any:
        """Plan, gather evidence, reason; returns a result with ``.output``."""
        from pydantic_ai.usage import UsageLimits

        plan = build_plan(request, deps)
        evidence, meta = build_evidence(plan, deps)
        prompt = (
            f"{evidence}\n\n"
            f"{request}\n"
            f"Output exactly {plan['k']} items, best first."
        )
        result = await self.agent.run(
            prompt,
            usage_limits=UsageLimits(request_limit=self.max_requests),
        )
        output = result.output
        items = _clean_items(output.items if output else [], plan=plan, meta=meta)
        return dataclasses.replace(result, output=RankedItems(items=items))

    def run(self, request: str, deps: ToolRegistry) -> Any:
        """Sync convenience wrapper for the CLI."""
        return asyncio.run(self.arun(request, deps))


def usage_summary(result: Any) -> dict:
    usage = result.usage() if callable(getattr(result, "usage", None)) else result.usage
    return {
        "requests": getattr(usage, "requests", 0),
        "prompt_tokens": getattr(
            usage, "input_tokens", getattr(usage, "request_tokens", 0)
        ),
        "completion_tokens": getattr(
            usage, "output_tokens", getattr(usage, "response_tokens", 0)
        ),
    }
