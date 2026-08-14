"""Typed Python client for a running ``recagent serve`` gateway.

Use it to wire the recommender into other systems:

    from recagent_sdk import RecommendClient

    client = RecommendClient("http://localhost:8000")
    resp = client.recommend(user_id=196, k=5, filters={"genre": "Sci-Fi"})
    for item in resp.items:
        print(item.title, item.reason)
"""

from __future__ import annotations

from typing import Any, Self

import httpx

from recagent_sdk.models import (
    CatalogEntry,
    ChatResponse,
    FeedbackResponse,
    HealthResponse,
    RecommendResponse,
)


class RecagentError(RuntimeError):
    """Raised when the gateway returns a non-2xx response."""


def _ensure(resp: httpx.Response, model: type[Any]) -> Any:
    if resp.status_code >= 400:
        raise RecagentError(
            f"recagent API error {resp.status_code}: {resp.text[:200]}"
        )
    return model.model_validate(resp.json())


class RecommendClient:
    """Typed, thread-safe client for the recagent REST gateway."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        kwargs: dict[str, Any] = {"base_url": self.base_url, "timeout": timeout}
        if transport is not None:
            kwargs["transport"] = transport
        self._http = httpx.Client(**kwargs)
        self._http_async = httpx.AsyncClient(**kwargs)

    # -- sync --------------------------------------------------------------

    def health(self) -> HealthResponse:
        return _ensure(self._http.get("/health"), HealthResponse)

    def recommend(
        self,
        user_id: int,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> RecommendResponse:
        return _ensure(
            self._http.post("/recommend", json={"user_id": user_id, "k": k, "filters": filters}),
            RecommendResponse,
        )

    def chat(self, message: str, *, user_id: int | None = None) -> ChatResponse:
        return _ensure(
            self._http.post("/chat", json={"message": message, "user_id": user_id}),
            ChatResponse,
        )

    def feedback(self, user_id: int, item_id: int, liked: bool) -> FeedbackResponse:
        return _ensure(
            self._http.post(
                "/feedback",
                json={"user_id": user_id, "item_id": item_id, "liked": liked},
            ),
            FeedbackResponse,
        )

    def catalog(self, item_id: int) -> CatalogEntry:
        return _ensure(self._http.get(f"/catalog/{item_id}"), CatalogEntry)

    # -- async --------------------------------------------------------------

    async def ahealth(self) -> HealthResponse:
        return _ensure(await self._http_async.get("/health"), HealthResponse)

    async def arecommend(
        self,
        user_id: int,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> RecommendResponse:
        resp = await self._http_async.post(
            "/recommend", json={"user_id": user_id, "k": k, "filters": filters}
        )
        return _ensure(resp, RecommendResponse)

    async def achat(self, message: str, *, user_id: int | None = None) -> ChatResponse:
        resp = await self._http_async.post(
            "/chat", json={"message": message, "user_id": user_id}
        )
        return _ensure(resp, ChatResponse)

    async def afeedback(self, user_id: int, item_id: int, liked: bool) -> FeedbackResponse:
        resp = await self._http_async.post(
            "/feedback",
            json={"user_id": user_id, "item_id": item_id, "liked": liked},
        )
        return _ensure(resp, FeedbackResponse)

    async def acatalog(self, item_id: int) -> CatalogEntry:
        resp = await self._http_async.get(f"/catalog/{item_id}")
        return _ensure(resp, CatalogEntry)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._http.close()
        self._http_async.aclose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
