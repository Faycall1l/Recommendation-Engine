"""Shared utilities for recagent."""

from __future__ import annotations

from typing import Any


def usage_summary(result: Any) -> dict[str, int]:
    """Extract token usage from a pydantic-ai result object."""
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
