"""recagent — an agentic recommender system."""

from recagent.agent import ReasoningTrace, RecAgent
from recagent.client import RecClient
from recagent.explain import RecExplainer
from recagent.tools import ToolRegistry

__version__ = "0.1.0"

__all__ = [
    "ReasoningTrace",
    "RecAgent",
    "RecClient",
    "RecExplainer",
    "ToolRegistry",
]
