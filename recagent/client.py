"""Typed client facade: the programmatic API other systems call directly.

One import surface for the whole pipeline: recommendation requests with
structured filters, free-form chat, item explanations, and feedback capture.
When no LLM endpoint is configured the client degrades gracefully to the raw
collaborative filter, so it is usable without any model weights.

Includes retry with exponential backoff for transient vLLM failures and a
circuit breaker that opens after repeated failures to avoid hammering a
struggling endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from recagent.agent import RecAgent, build_evidence, build_plan
from recagent.config import LLMConfig, load_llm_config
from recagent.explain import Explanation, RecExplainer, explain_recommendation
from recagent.memory import UserMemory
from recagent.session import SessionMemory
from recagent.state import load_state
from recagent.tools import ToolRegistry
from recagent.utils import usage_summary

logger = logging.getLogger(__name__)


class Recommendation(BaseModel):
    item_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    score: float | None = None
    reason: str = ""
    tags: list[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    user_id: int
    k: int
    items: list[Recommendation]
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float | None = None
    reflection_applied: bool | None = None


class ChatResponse(BaseModel):
    user_id: int | None
    items: list[Recommendation]
    evidence: str = ""
    usage: dict[str, int] = Field(default_factory=dict)


class ExplanationResponse(BaseModel):
    user_id: int
    explanation: Explanation
    text: str
    llm: bool
    usage: dict[str, int] = Field(default_factory=dict)


class FeedbackEvent(BaseModel):
    user_id: int
    item_id: int
    liked: bool


class _CircuitBreaker:
    """Thread-safe fail-fast guard for a flaky endpoint.

    States:
      CLOSED  — normal operation; consecutive failures counted.
      OPEN    — after ``threshold`` consecutive failures, all calls are
                rejected immediately for ``reset_timeout`` seconds.
      HALF_OPEN — after the timeout, one trial call is allowed; success
                  closes the circuit, failure reopens it.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, threshold: int = 3, reset_timeout: float = 30.0):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN and time.monotonic() - self._opened_at >= self.reset_timeout:
                self._state = self.HALF_OPEN
            return self._state

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "circuit breaker opened after %d consecutive failures", self._failures
                )

    def allow_request(self) -> bool:
        st = self.state
        return st in (self.CLOSED, self.HALF_OPEN)


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
                tags=getattr(item, "tags", []),
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
        explainer: RecExplainer | None = None,
        llm_config: LLMConfig | None = None,
        feedback_path: str | Path | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        circuit_threshold: int = 3,
        circuit_timeout: float = 30.0,
    ):
        self.state = state if state is not None else load_state(str(artifacts))
        self.feedback_path = (
            Path(feedback_path) if feedback_path else Path(str(artifacts)) / "feedback.jsonl"
        )
        self.memory = UserMemory(Path(str(artifacts)) / "memory.json")
        self.session = SessionMemory()
        self.deps = ToolRegistry(self.state, memory=self.memory, session=self.session)
        # auto-ingest any existing feedback into memory buckets
        self.memory.ingest_feedback(self.feedback_path)
        config = llm_config or load_llm_config()
        self.agent = agent if agent is not None else (RecAgent(config, self.state) if config.enabled else None)
        self.explainer = (
            explainer
            if explainer is not None
            else (RecExplainer(config) if config.enabled else None)
        )
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._breaker = _CircuitBreaker(threshold=circuit_threshold, reset_timeout=circuit_timeout)

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

    def _run_with_retry(self, request: str, *, request_id: str = "") -> Any:
        """Run the agent with exponential backoff retry + circuit breaker.

        Only retries on transient errors (connection, timeout, transport).
        Permanent errors (4xx, auth) are raised immediately.
        """
        import random

        import httpx

        transient = (httpx.TransportError, ConnectionError, TimeoutError, OSError)

        if not self._breaker.allow_request():
            raise ConnectionError("circuit breaker open — LLM endpoint unreachable")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.agent.run(request, self.deps)
                self._breaker.record_success()
                return result
            except transient as exc:
                last_exc = exc
                self._breaker.record_failure()
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2**attempt) * (0.5 + random.random() * 0.5)
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            except Exception:
                raise
        raise last_exc  # type: ignore[misc]

    async def _arun_with_retry(self, request: str, *, request_id: str = "") -> Any:
        """Async run the agent with exponential backoff retry + circuit breaker.

        Only retries on transient errors (connection, timeout, transport).
        """
        import random

        import httpx

        transient = (httpx.TransportError, ConnectionError, TimeoutError, OSError)

        if not self._breaker.allow_request():
            raise ConnectionError("circuit breaker open — LLM endpoint unreachable")
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await self.agent.arun(request, self.deps, request_id=request_id)
                self._breaker.record_success()
                return result
            except transient as exc:
                last_exc = exc
                self._breaker.record_failure()
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2**attempt) * (0.5 + random.random() * 0.5)
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
            except Exception:
                raise
        raise last_exc  # type: ignore[misc]

    # -- public API ----------------------------------------------------------

    def recommend(
        self, user_id: int, k: int = 5, filters: dict[str, Any] | None = None,
        *, request_id: str | None = None,
    ) -> RecommendResponse:
        """Recommend ``k`` items for a user.

        ``filters`` supports structured constraints, e.g. ``{"genre": "Sci-Fi"}``.
        ``request_id`` is threaded through to the agent trace for correlation.
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
            resp = RecommendResponse(user_id=user_id, k=k, items=items)
            self.session.record_recommendation(request, [r.item_id for r in resp.items])
            return resp
        result = self._run_with_retry(request, request_id=request_id)
        trace = getattr(result, "trace", None)
        resp = RecommendResponse(
            user_id=user_id,
            k=k,
            items=self._from_agent(result, user_id),
            usage=usage_summary(result),
            latency_ms=trace.latency_ms if trace else None,
            reflection_applied=trace.refinement_applied if trace else None,
        )
        self.session.record_recommendation(request, [r.item_id for r in resp.items])
        logger.info(
            "recommend user_id=%d k=%d items=%d latency_ms=%.1f reflection=%s",
            user_id, k, len(resp.items), resp.latency_ms or 0, resp.reflection_applied,
        )
        return resp

    async def arecommend(
        self, user_id: int, k: int = 5, filters: dict[str, Any] | None = None,
        *, request_id: str | None = None,
    ) -> RecommendResponse:
        request = self._filters_request(user_id, k, filters)
        if self.agent is None:
            return await asyncio.to_thread(self.recommend, user_id, k, filters)
        result = await self._arun_with_retry(request, request_id=request_id)
        trace = getattr(result, "trace", None)
        resp = RecommendResponse(
            user_id=user_id,
            k=k,
            items=self._from_agent(result, user_id),
            usage=usage_summary(result),
            latency_ms=trace.latency_ms if trace else None,
            reflection_applied=trace.refinement_applied if trace else None,
        )
        self.session.record_recommendation(request, [r.item_id for r in resp.items])
        return resp

    def chat(
        self, message: str, *, user_id: int | None = None, k: int = 5
    ) -> ChatResponse:
        """Free-form request; bind to a user with ``user_id``."""
        request = message if user_id is None else f"user_id: {user_id}\n\n{message}"
        if self.agent is None:
            return ChatResponse(
                user_id=user_id, items=[], evidence="agent disabled — CF only", usage={}
            )
        result = self._run_with_retry(request)
        items = self._from_agent(result, user_id or 0)
        plan = build_plan(request, self.deps)
        evidence, _, _ = build_evidence(plan, self.deps)
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
        result = await self._arun_with_retry(request)
        items = self._from_agent(result, user_id or 0)
        plan = build_plan(request, self.deps)
        evidence, _, _ = build_evidence(plan, self.deps)
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

    def explain_recommendation(
        self, user_id: int, item_id: int, *, recommended_ids: set[int] | None = None
    ) -> ExplanationResponse:
        """Why this item for this user: deterministic evidence + grounded prose.

        The LLM restates the evidence when enabled; otherwise the deterministic
        snippet stands in, so an explanation always exists.
        When ``recommended_ids`` is provided the explanation includes a
        contrastive comparison against the best alternative.
        """
        explanation = explain_recommendation(
            self.deps, user_id, item_id, recommended_ids=recommended_ids
        )
        if self.explainer is None:
            return ExplanationResponse(
                user_id=user_id, explanation=explanation, text=explanation.snippet, llm=False
            )
        text, usage = self.explainer.explain(explanation)
        return ExplanationResponse(
            user_id=user_id, explanation=explanation, text=text, llm=True, usage=usage
        )

    async def aexplain_recommendation(
        self, user_id: int, item_id: int, *, recommended_ids: set[int] | None = None
    ) -> ExplanationResponse:
        explanation = explain_recommendation(
            self.deps, user_id, item_id, recommended_ids=recommended_ids
        )
        if self.explainer is None:
            return ExplanationResponse(
                user_id=user_id, explanation=explanation, text=explanation.snippet, llm=False
            )
        text, usage = await self.explainer.aexplain(explanation)
        return ExplanationResponse(
            user_id=user_id, explanation=explanation, text=text, llm=True, usage=usage
        )

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
