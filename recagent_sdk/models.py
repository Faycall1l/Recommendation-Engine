"""Response models for the recagent REST API — shared contract, no core imports."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class FeedbackResponse(BaseModel):
    accepted: bool
    event: dict = Field(default_factory=dict)


class CatalogEntry(BaseModel):
    item_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    rating_count: int = 0
    avg_rating: float | None = None


class Explanation(BaseModel):
    item_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    score: float | None = None
    user_mean: float | None = None
    boost: float | None = None
    user_top_genres: list[str] = Field(default_factory=list)
    matched_genres: list[str] = Field(default_factory=list)
    user_likes: list[dict] = Field(default_factory=list)
    similar_rated: list[dict] = Field(default_factory=list)
    avg_rating: float | None = None
    rating_count: int = 0
    basis: str = ""
    snippet: str = ""


class ExplainResponse(BaseModel):
    user_id: int
    explanation: Explanation
    text: str
    llm: bool
    usage: dict[str, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    agent_enabled: bool
    model: str | None = None
    n_users: int
    n_items: int
