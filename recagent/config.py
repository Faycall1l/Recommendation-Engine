"""Runtime configuration, sourced from .env / environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

_BASE_URL_DEFAULT = "http://localhost:8000/v1"
_MODEL_DEFAULT = "Gemma-4-31B-it"


class LLMConfig(BaseModel):
    """Connection settings for the OpenAI-compatible vLLM endpoint."""

    enabled: bool = False
    base_url: str = _BASE_URL_DEFAULT
    api_key: str = ""
    model: str = _MODEL_DEFAULT

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with http:// or https://, got {v!r}")
        return v

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model must be a non-empty string")
        return v

    def model_post_init(self, __context: object, /) -> None:
        if self.enabled and not self.api_key:
            raise ValueError("api_key is required when enabled=True")


class RecAgentConfig(BaseModel):
    """Agent behaviour tuning — controls the plan-reflect-diversify pipeline."""

    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_requests: int = Field(default=12, ge=1, le=100)
    reflect: bool = True
    diversity: bool = True
    lambda_param: float = Field(default=0.5, ge=0.0, le=1.0,
                                description="MMR relevance/diversity tradeoff: 1.0=pure relevance, 0.0=pure diversity")
    evidence_budget_tokens: int = Field(default=4000, ge=256, le=32000,
                                        description="Max characters for the evidence block sent to the LLM")


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        enabled=os.getenv("ATHAR_AGENT__ENABLED", "false").strip().lower()
        in {"1", "true", "yes"},
        base_url=os.getenv("ATHAR_AGENT__VLLM__BASE_URL", _BASE_URL_DEFAULT),
        api_key=os.getenv("ATHAR_AGENT__VLLM__API_KEY", ""),
        model=os.getenv("ATHAR_AGENT__VLLM__MODEL", _MODEL_DEFAULT),
    )
