"""Runtime configuration, sourced from .env / environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_BASE_URL_DEFAULT = "http://localhost:8000/v1"
_MODEL_DEFAULT = "Gemma-4-31B-it"


@dataclass
class LLMConfig:
    """Connection settings for the OpenAI-compatible vLLM endpoint."""

    enabled: bool = False
    base_url: str = _BASE_URL_DEFAULT
    api_key: str = ""
    model: str = _MODEL_DEFAULT


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        enabled=os.getenv("ATHAR_AGENT__ENABLED", "false").strip().lower()
        in {"1", "true", "yes"},
        base_url=os.getenv("ATHAR_AGENT__VLLM__BASE_URL", _BASE_URL_DEFAULT),
        api_key=os.getenv("ATHAR_AGENT__VLLM__API_KEY", ""),
        model=os.getenv("ATHAR_AGENT__VLLM__MODEL", _MODEL_DEFAULT),
    )
