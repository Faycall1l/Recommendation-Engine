"""Typed client facade: the programmatic API other systems call directly.

One import surface for the whole pipeline: recommendation requests with
structured filters, free-form chat, item explanations, and feedback capture.
When no LLM endpoint is configured the client degrades gracefully to the raw
collaborative filter, so it is usable without any model weights.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from recagent.agent import RecAgent, build_evidence, build_plan, usage_summary
from recagent.config import LLMConfig, load_llm_config
from recagent.state import load_state
from recagent.tools import ToolRegistry


class Recommendation(BaseModel):
    item_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    score: float | None = None
    reason: str = ""


class RecommendResponse(BaseModel):
    user_id: int
    k: int
    items: list[Recommendation]
    usage: dict[str, int] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    user_id: int | None
    items: list[Recommendation]
    evidence: str = ""
    usage: dict[str, int] = Field(default_factory=dict)


class FeedbackEvent(BaseModel):
    user_id: int
    item_id: int
    liked: bool


def _to_recommendations(
    ranked: Any, deps: ToolRegistry, *, scores: dict[int, float] | None = None
) -> list[Recommendation]:
    """Map agent output to the public schema, filling catalog metadata."""
    out: list[Recommendation] = []
    for item in ranked.items:
        info = deps.items_meta.get(item.item_id, {})
        out.append(
            Recommendation(
                item_id=item.item_id,
                title=info.get("title", f"<item {item.item_id}>"),
                genres=info.get("genres", []),
                score=(scores or {}).get(item.item_id),
                reason=item.reason,
            )
        )
    return out


class RecClient:
    """Thread-safe facade over the pipeline; async variants use ``await``."""

    def __init__(
        self,
        artifacts: str | Path = "artifacts",
        *,
        state: dict[str, Any] | None = None,
        agent: RecAgent | None = None,
        llm_config: LLMConfig | None = None,
        feedback_path: str | Path | None = None,
    ):
        self.state = state if state is not None else load_state(str(artifacts))
        self.deps = ToolRegistry(self.state)
        if agent is not None:
            self.agent = agent
        else:
            config = llm_config or load_llm_config()
            self.agent = RecAgent(config, self.state) if config.enabled else None
        self.feedback_path = (
            Path(feedback_path) if feedback_path else Path(str(artifacts)) / "feedback.jsonl"
        )

    # -- low-level ----------------------------------------------------------

    def _cf_scores(self, user_id: int) -> dict[int, float]:
        if user_id not in self.deps.uid_to_idx:
            return {}
        return {e.item_id: e.score for e in self.deps.recommend(user_id, n=50).items}

    def _filters_request(self, user_id: int, k: int, filters: dict[str, Any] | None) -> str:
        parts = [f"Recommend {k} items for user_id: {user_id}."]
        genre = (filters or {}).get("genre")
        if genre:
            parts.append(f"Constraint: every item must be {genre}.")
        return " ".join(parts)

    def _from_agent(self, result: Any, user_id: int) -> list[Recommendation]:
        return _to_recommendations(
            result.output or type(result.output)(items=[]),
            self.deps,
            scores=self._cf_scores(user_id),
        )

    # -- public API ----------------------------------------------------------

    def recommend(
        self, user_id: int, k: int = 5, filters: dict[str, Any] | None = None
    ) -> RecommendResponse:
        """Recommend ``k`` items for a user.

        ``filters`` supports structured constraints, e.g. ``{"genre": "Sci-Fi"}``.
        """
        request = self._filters_request(user_id, k, filters)
        if self.agent is None:
            if user_id not in self.deps.uid_to_idx:
                items = [
                    Recommendation(
                        item_id=e.item_id,
                        title=e.title,
                        genres=e.genres,
                        score=e.score,
                        reason="popularity prior (cold start)",
                    )
                    for e in self.deps.trending(n=k).items
                ]
            else:
                items = [
                    Recommendation(
                        item_id=e.item_id,
                        title=e.title,
                        genres=e.genres,
                        score=e.score,
                        reason="collaborative filter",
                    )
                    for e in self.deps.recommend(user_id, n=k).items
                ]
            return RecommendResponse(user_id=user_id, k=k, items=items)
        result = self.agent.run(request, self.deps)
        return RecommendResponse(
            user_id=user_id,
            k=k,
            items=self._from_agent(result, user_id),
            usage=usage_summary(result),
        )

    async def arecommend(
        self, user_id: int, k: int = 5, filters: dict[str, Any] | None = None
    ) -> RecommendResponse:
        request = self._filters_request(user_id, k, filters)
        if self.agent is None:
            return await asyncio.to_thread(self.recommend, user_id, k, filters)
        result = await self.agent.arun(request, self.deps)
        return RecommendResponse(
            user_id=user_id,
            k=k,
            items=self._from_agent(result, user_id),
            usage=usage_summary(result),
        )

    def chat(
        self, message: str, *, user_id: int | None = None, k: int = 5
    ) -> ChatResponse:
        """Free-form request; bind to a user with ``user_id``."""
        request = message if user_id is None else f"user_id: {user_id}\n\n{message}"
        if self.agent is None:
            return ChatResponse(
                user_id=user_id, items=[], evidence="agent disabled — CF only", usage={}
            )
        result = self.agent.run(request, self.deps)
        items = self._from_agent(result, user_id or 0)
        plan = build_plan(request, self.deps)
        evidence, _ = build_evidence(plan, self.deps)
        return ChatResponse(
            user_id=user_id,
            items=items,
            evidence=evidence,
            usage=usage_summary(result),
        )

    async def achat(
        self, message: str, *, user_id: int | None = None, k: int = 5
    ) -> ChatResponse:
        request = message if user_id is None else f"user_id: {user_id}\n\n{message}"
        if self.agent is None:
            return ChatResponse(
                user_id=user_id, items=[], evidence="agent disabled — CF only", usage={}
            )
        result = await self.agent.arun(request, self.deps)
        items = self._from_agent(result, user_id or 0)
        plan = build_plan(request, self.deps)
        evidence, _ = build_evidence(plan, self.deps)
        return ChatResponse(
            user_id=user_id,
            items=items,
            evidence=evidence,
            usage=usage_summary(result),
        )

    def explain(self, item_id: int) -> dict[str, Any]:
        """Catalog metadata + popularity for one item."""
        entry = self.deps.item_info(item_id)
        return {
            "item_id": entry.item_id,
            "title": entry.title,
            "genres": entry.genres,
            "rating_count": entry.rating_count,
            "avg_rating": entry.avg_rating,
        }

    def feedback(self, user_id: int, item_id: int, liked: bool) -> dict[str, Any]:
        """Record an explicit like/dislike; persisted to a JSONL sidecar."""
        event = FeedbackEvent(user_id=user_id, item_id=item_id, liked=liked)
        try:
            with self.feedback_path.open("a") as fh:
                fh.write(event.model_dump_json() + "\n")
            stored = True
        except OSError:
            stored = False
        return {"accepted": stored, "event": event.model_dump()}

    def health(self) -> dict[str, Any]:
        matrix = self.state["matrix"]
        return {
            "status": "ok",
            "agent_enabled": self.agent is not None,
            "model": self.agent.config.model if self.agent else None,
            "n_users": int(matrix.shape[0]),
            "n_items": int(matrix.shape[1]),
        }
