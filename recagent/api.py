"""REST gateway: the same pipeline behind HTTP, for other systems to wire in.

The service owns one :class:`RecClient` and exposes typed JSON endpoints that
mirror the client facade. Swagger docs land at ``/docs``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from recagent.client import RecClient


class RecommendRequest(BaseModel):
    user_id: int
    k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    message: str
    user_id: int | None = None


class ExplainRequest(BaseModel):
    user_id: int
    item_id: int


class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int
    liked: bool


def create_app(
    artifacts: str | Path = "artifacts",
    client: RecClient | None = None,
) -> FastAPI:
    """Build the FastAPI app; inject ``client`` for tests."""
    rec = client or RecClient(artifacts)

    app = FastAPI(
        title="recagent",
        version="0.1.0",
        description="Agentic recommender: collaborative-filter candidates, refined by an "
        "LLM with evidence-grounded reasoning.",
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return rec.health()

    @app.post("/recommend")
    async def recommend(req: RecommendRequest) -> Any:
        return await rec.arecommend(req.user_id, k=req.k, filters=req.filters)

    @app.post("/chat")
    async def chat(req: ChatRequest) -> Any:
        return await rec.achat(req.message, user_id=req.user_id)

    @app.post("/explain")
    async def explain(req: ExplainRequest) -> Any:
        return await rec.aexplain_recommendation(req.user_id, req.item_id)

    @app.post("/feedback")
    async def feedback(req: FeedbackRequest) -> dict[str, Any]:
        return rec.feedback(req.user_id, req.item_id, req.liked)

    @app.get("/catalog/{item_id}")
    async def catalog(item_id: int) -> dict[str, Any]:
        return rec.explain(item_id)

    return app
