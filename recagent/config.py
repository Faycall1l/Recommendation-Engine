"""Runtime configuration, sourced from .env / environment variables."""

from __future__ import annotations

import os

from pydantic import BaseModel, field_validator

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


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        enabled=os.getenv("ATHAR_AGENT__ENABLED", "false").strip().lower()
        in {"1", "true", "yes"},
        base_url=os.getenv("ATHAR_AGENT__VLLM__BASE_URL", _BASE_URL_DEFAULT),
        api_key=os.getenv("ATHAR_AGENT__VLLM__API_KEY", ""),
        model=os.getenv("ATHAR_AGENT__VLLM__MODEL", _MODEL_DEFAULT),
    )
