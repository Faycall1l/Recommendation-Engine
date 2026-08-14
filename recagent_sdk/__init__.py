"""recagent-sdk: wire any system into the recagent recommendation gateway."""

from recagent_sdk.client import RecagentError, RecommendClient
from recagent_sdk.models import (
    CatalogEntry,
    ChatResponse,
    ExplainResponse,
    FeedbackResponse,
    HealthResponse,
    Recommendation,
    RecommendResponse,
)

__all__ = [
    "CatalogEntry",
    "ChatResponse",
    "ExplainResponse",
    "FeedbackResponse",
    "HealthResponse",
    "RecagentError",
    "RecommendClient",
    "RecommendResponse",
    "Recommendation",
]
