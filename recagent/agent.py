"""The agentic recommender: a plan-then-execute agent over a vLLM-served Gemma-4.

Plan    — the harness parses the request (user, item count, any genre constraint).
Execute — the harness gathers evidence through the CF tool registry: the user's
          taste profile, engine-ranked candidates, and constraint-matched items.
Reason  — one structured call to the LLM produces a typed, grounded ranked list.
Reflect — a deterministic self-check catches constraint violations, low diversity,
          or poor evidence coverage; when issues are found a second targeted LLM
          call fixes them before final cleaning.

Tools stay first-class (they are planned and executed); running them from a
deterministic harness instead of in the model's loop is dramatically more
reliable against the vLLM tool-calling template, and it is still scored on the
exact lists the agent emits.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from recagent.config import LLMConfig
from recagent.tools import ToolRegistry
from recagent.utils import usage_summary


class RankedItem(BaseModel):
    item_id: int = Field(description="item id from the catalog")
    reason: str = Field(description="one-sentence justification grounded in evidence")


class RankedItems(BaseModel):
    items: list[RankedItem] = Field(description="ranked list, best first")


@dataclasses.dataclass(frozen=True, slots=True)
class ReflectionReport:
    """Deterministic post-LLM quality check."""

    issues: list[str]
    needs_refinement: bool


@dataclasses.dataclass(slots=True)
class ReasoningTrace:
    """Machine-readable audit log of one agent pipeline execution.

    Captures every stage so production debugging is possible without
    replaying the entire request. Returned by ``RecAgent.arun()`` as the
    ``.trace`` attribute on the result.
    """

    request_id: str = ""
    plan_user_id: int | None = None
    plan_k: int = 5
    plan_constraint: str | None = None
    evidence_text: str = ""
    evidence_sections: list[str] = dataclasses.field(default_factory=list)
    evidence_meta: dict[str, object] = dataclasses.field(default_factory=dict)
    raw_llm_output: str = ""
    reflection_issues: list[str] = dataclasses.field(default_factory=list)
    refinement_applied: bool = False
    cleaned_item_ids: list[int] = dataclasses.field(default_factory=list)
    diversity_applied: bool = False
    latency_ms: float = 0.0
    usage: dict[str, object] = dataclasses.field(default_factory=dict)


SYSTEM_PROMPT = """\
You are RecAgent, a recommendation agent with persistent memory. You produce
ranked item lists grounded in evidence: user taste profile, collaborative
filtering candidates, preference history, and social proof.

You think about the user holistically:
- What they have liked and disliked before (from preference buckets)
- What similar users enjoy (social proof)
- How the request differs from their usual taste (mood, context, discovery)
- Why one item over another (contrastive reasoning)

Rules:
- Choose items only from the evidence block. Never invent item ids.
- Honor explicit constraints. If the request says every item must be a genre,
  every output item must carry that genre.
- Rank best first: blend the engine score with fit to the user's taste and the
  constraint. Prefer items with strong popularity and average rating when scores
  are close; items liked by the user's most similar neighbours are strong fits.
- When preference history is available, use it: avoid items similar to
  disliked ones, lean toward genres in the loved bucket, and vary from
  recent recommendations to avoid repetition.
- For mood or context requests (e.g. "something light", "late night watch"),
  interpret the intent and adjust the ranking accordingly.
- For discovery requests (e.g. "surprise me", "something different"), favour
  items outside the user's dominant genres but still relevant to their taste.
- Justify each pick in one short sentence using the evidence.
- Do not repeat an item. Output as many items as requested, but fewer is better
  than padding with items that violate a constraint or are not in the evidence.
"""


def _genre_aliases(genre: str) -> list[str]:
    """Lowercased search forms for a catalog genre label."""
    base = genre.lower()
    if base == "sci-fi":
        return ["sci-fi", "sci fi", "scifi", "science fiction"]
    if base == "film-noir":
        return ["film-noir", "film noir", "noir"]
    if base == "children's":
        return ["children's", "children", "kids"]
    return [base]


def detect_genre(request: str, deps: ToolRegistry) -> str | None:
    """The catalog genre a free-text request refers to, if any."""
    haystack = request.lower()
    genres = {g for info in deps.items_meta.values() for g in info.get("genres", [])}
    for genre in sorted(genres):
        if any(alias in haystack for alias in _genre_aliases(genre)):
            return genre
    return None


def build_plan(request: str, deps: ToolRegistry, default_k: int = 5) -> dict[str, Any]:
    """Parse the request into a deterministic execution plan."""
    user_match = re.search(r"user_id:\s*(\d+)", request)
    if not user_match:
        raise ValueError(f"no user_id in request: {request!r}")
    k_match = re.search(r"(\d+)\s*items", request)
    return {
        "user_id": int(user_match.group(1)),
        "k": int(k_match.group(1)) if k_match else default_k,
        "genre": detect_genre(request, deps),
    }


def _fmt(entry: Any, detail: str | None = None) -> str:
    genres = ", ".join(entry.genres) or "-"
    parts = [f"- {entry.item_id}: {entry.title} [{genres}]"]
    if entry.rating_count:
        parts.append(f"watched {entry.rating_count}x")
    if entry.avg_rating is not None:
        parts.append(f"avg {entry.avg_rating:.1f}")
    if detail:
        parts.append(detail)
    return ", ".join(parts)


def build_evidence(
    plan: dict[str, Any], deps: ToolRegistry, *, budget_tokens: int = 4000
) -> tuple[str, dict[int, set[str]], dict[str, list[int]]]:
    """Gather evidence through the tools; return (text block, item genre map, evidence sections).

    Adaptive retrieval: warm users get profile + CF candidates + social proof;
    cold users fall back to a popularity prior; a rare genre widens the pool
    through item-item neighbours of the genre hits. Every item carries its
    popularity (times watched) and average rating so the model can weigh quality
    against raw score.

    The third return value maps section names to the item_ids they contributed,
    used by the reflection step to check evidence coverage.

    ``budget_tokens`` caps the evidence text at roughly that many tokens
    (1 token ~ 4 chars) to avoid flooding the LLM context window.
    """
    meta: dict[int, set[str]] = {}
    sections: dict[str, list[int]] = {}
    uid, k = plan["user_id"], plan["k"]
    genre = plan.get("genre")

    def absorb(items: list[Any], section: str) -> None:
        ids: list[int] = []
        for entry in items:
            meta[entry.item_id] = set(entry.genres)
            ids.append(entry.item_id)
        sections[section] = ids

    lines: list[str] = []
    if uid in deps.uid_to_idx:
        stats = deps.user_stats(uid)
        lines.append(
            f"User rating context (user_id: {uid}): {stats.n_rated} movies rated, "
            f"mean rating {stats.mean_rating:.2f} (1-5 scale)."
        )
        profile = deps.user_profile(uid, k=min(8, max(4, k)))
        lines.append("User profile (highest rated):")
        for entry in profile.items:
            lines.append(_fmt(entry, detail=f"rating {entry.rating}"))
        absorb(profile.items, "profile")

        candidates = deps.recommend(uid, n=20)
        lines.append(
            "Collaborative filtering candidates (engine score, best first; "
            "scores are comparable within this list, higher is stronger):"
        )
        for entry in candidates.items:
            lines.append(_fmt(entry, detail=f"score {entry.score}"))
        absorb(candidates.items, "candidates")

        social = deps.similar_users(uid, n=3)
        if social.users:
            lines.append("Social proof (what the most similar users like):")
            social_ids: list[int] = []
            for peer in social.users:
                peer_profile = deps.user_profile(peer.user_id, k=3)
                for entry in peer_profile.items:
                    lines.append(
                        _fmt(entry, detail=f"liked by similar user {peer.user_id}")
                    )
                    meta[entry.item_id] = set(entry.genres)
                    social_ids.append(entry.item_id)
            sections["social"] = social_ids
    else:
        lines.append(f"Cold-start user (user_id: {uid}) — no interaction history.")
        prior = deps.trending(n=20)
        lines.append("Popularity prior (most-watched across all users):")
        for entry in prior.items:
            lines.append(_fmt(entry, detail=f"rated {entry.rating_count}x"))
        absorb(prior.items, "trending")

    if genre:
        hits = deps.search_items(genre, n=15)
        lines.append(f"Search matches for the requested genre ({genre}):")
        for entry in hits.items:
            lines.append(_fmt(entry))
        absorb(hits.items, "genre_search")

        if len(hits.items) < k:
            widened: dict[int, Any] = {}
            for entry in hits.items:
                for neighbour in deps.similar_items(entry.item_id, n=3).items:
                    widened.setdefault(neighbour.item_id, neighbour)
            if widened:
                lines.append("Widened candidates (similar to the genre hits):")
                widened_items = list(widened.values())[:15]
                for entry in widened_items:
                    lines.append(_fmt(entry, detail=f"similarity {entry.score}"))
                absorb(widened_items, "genre_widened")

    text = "\n".join(lines)
    max_chars = budget_tokens * 4
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...evidence truncated to budget...]"
    return text, meta, sections


def _clean_items(
    items: list[RankedItem],
    *,
    plan: dict[str, Any],
    meta: dict[int, set[str]],
    diversity: bool = True,
    lambda_param: float = 0.5,
) -> list[RankedItem]:
    """Drop hallucinations, constraint violations, and duplicates; optionally
    apply MMR re-ranking for genre diversity; cap at k."""
    seen: set[int] = set()
    kept: list[RankedItem] = []
    genre = plan.get("genre")
    for item in items:
        genres = meta.get(item.item_id)
        if genres is None or item.item_id in seen:
            continue
        if genre and genre not in genres:
            continue
        seen.add(item.item_id)
        kept.append(item)
    if diversity and len(kept) > 1:
        kept = mmr_rerank(kept, meta, lambda_param=lambda_param)
    return kept[: plan["k"]]


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two genre sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def mmr_rerank(
    items: list[RankedItem],
    meta: dict[int, set[str]],
    *,
    lambda_param: float = 0.5,
) -> list[RankedItem]:
    """Maximal Marginal Relevance re-ranking for genre diversity.

    The first item (LLM's top pick) is always selected. Each subsequent item
    balances relevance (original rank position) against diversity (dissimilarity
    from already-selected items), using Jaccard similarity on genre sets.

    ``lambda_param`` controls the tradeoff: 1.0 = pure relevance (no diversity
    benefit), 0.0 = pure diversity (ignores rank order).
    """
    if len(items) <= 1:
        return items
    n = len(items)
    # relevance: inverse of position (first item = 1.0, last = 1/n)
    relevance = [1.0 - i / n for i in range(n)]
    genre_sets = [meta.get(it.item_id, set()) for it in items]

    selected_idx: list[int] = [0]
    remaining = set(range(1, n))

    while remaining and len(selected_idx) < n:
        best_score = -1.0
        best_j = -1
        for j in remaining:
            # max similarity to any already-selected item
            max_sim = max(
                _jaccard(genre_sets[j], genre_sets[s]) for s in selected_idx
            )
            score = lambda_param * relevance[j] - (1 - lambda_param) * max_sim
            if score > best_score:
                best_score = score
                best_j = j
        selected_idx.append(best_j)
        remaining.discard(best_j)

    return [items[i] for i in selected_idx]


def _reflect_on_ranking(
    raw_items: list[RankedItem],
    *,
    plan: dict[str, Any],
    meta: dict[int, set[str]],
    evidence_sections: dict[str, list[int]],
) -> ReflectionReport:
    """Deterministic post-LLM quality check.

    Inspects the raw (pre-clean) ranking for four classes of issues and
    returns a report; when ``needs_refinement`` is True the agent should
    issue a second LLM call with the issues embedded in the prompt.
    """
    issues: list[str] = []
    genre = plan.get("genre")
    k = plan["k"]

    cleaned = _clean_items(raw_items, plan=plan, meta=meta)

    # 1. Output too short after cleaning — the LLM likely violated constraints
    if len(cleaned) < k:
        issues.append(
            f"Only {len(cleaned)} of {k} requested items survive cleaning. "
            f"Review the evidence block and ensure every item matches the "
            f"{'genre constraint' if genre else 'request'}."
        )

    # 2. Raw constraint violation count (catches the problem source)
    if genre:
        violations = sum(
            1
            for it in raw_items
            if it.item_id in meta and genre not in meta[it.item_id]
        )
        if violations > 0:
            issues.append(
                f"{violations} of {len(raw_items)} raw items violate the "
                f"'{genre}' constraint. Every output item must carry '{genre}'."
            )

    # 3. Evidence coverage — the LLM should draw from multiple evidence sections
    if evidence_sections and len(evidence_sections) > 1 and cleaned:
        covered: set[str] = set()
        for it in cleaned:
            for section, ids in evidence_sections.items():
                if it.item_id in ids:
                    covered.add(section)
        uncovered = set(evidence_sections) - covered
        if uncovered:
            issues.append(
                f"The ranking uses no items from: {', '.join(sorted(uncovered))}. "
                f"Include items from these evidence sections for better coverage."
            )

    # 4. Genre diversity — all surviving items share the exact same genre set
    if len(cleaned) > 1:
        genre_sets = [meta.get(it.item_id, set()) for it in cleaned]
        if all(gs == genre_sets[0] for gs in genre_sets[1:]):
            issues.append(
                f"All {len(cleaned)} recommended items share identical genre(s): "
                f"{genre_sets[0]}. Include more diverse items from the evidence."
            )

    return ReflectionReport(issues=issues, needs_refinement=bool(issues))


class RecAgent:
    """Plan-then-execute recommender agent backed by an OpenAI-compatible vLLM model."""

    def __init__(
        self,
        config: LLMConfig,
        state: dict[str, Any],
        *,
        temperature: float = 0.1,
        max_requests: int = 12,
        reflect: bool = True,
        evidence_budget_tokens: int = 4000,
    ):
        self.config = config
        self.state = state
        self.max_requests = max_requests
        self.reflect = reflect
        self.evidence_budget_tokens = evidence_budget_tokens
        model = OpenAIChatModel(
            config.model,
            provider=OpenAIProvider(base_url=config.base_url, api_key=config.api_key),
        )
        self.agent = Agent(
            model,
            name="recagent",
            output_type=RankedItems,
            system_prompt=SYSTEM_PROMPT,
            retries=2,
            defer_model_check=True,
            model_settings=ModelSettings(
                temperature=temperature, max_tokens=2000, parallel_tool_calls=False
            ),
        )

    async def arun(self, request: str, deps: ToolRegistry, *, request_id: str = "") -> Any:
        """Plan, gather evidence, reason, optionally reflect; returns a result with ``.output``."""
        import time

        from pydantic_ai.usage import UsageLimits

        trace = ReasoningTrace(request_id=request_id)
        t0 = time.monotonic()

        plan = build_plan(request, deps)
        trace.plan_user_id = plan.get("user_id")
        trace.plan_k = plan.get("k", 5)
        trace.plan_constraint = plan.get("constraint")

        evidence, meta, evidence_sections = build_evidence(
            plan, deps, budget_tokens=self.evidence_budget_tokens
        )
        trace.evidence_text = evidence
        trace.evidence_sections = evidence_sections
        trace.evidence_meta = meta

        prompt = (
            f"{evidence}\n\n"
            f"{request}\n"
            f"Output exactly {plan['k']} items, best first."
        )
        result = await self.agent.run(
            prompt,
            usage_limits=UsageLimits(request_limit=self.max_requests),
        )
        raw_items = result.output.items if result.output else []
        trace.raw_llm_output = str(raw_items)

        if self.reflect:
            reflection = _reflect_on_ranking(
                raw_items,
                plan=plan,
                meta=meta,
                evidence_sections=evidence_sections,
            )
            trace.reflection_issues = list(reflection.issues)
            if reflection.needs_refinement:
                feedback = "\n".join(
                    f"Issue {i + 1}: {issue}"
                    for i, issue in enumerate(reflection.issues)
                )
                refinement_prompt = (
                    f"{evidence}\n\n"
                    f"Your initial ranking had the following issues:\n"
                    f"{feedback}\n\n"
                    f"{request}\n"
                    f"Output exactly {plan['k']} items, best first. "
                    f"Fix the issues listed above."
                )
                result = await self.agent.run(
                    refinement_prompt,
                    usage_limits=UsageLimits(request_limit=self.max_requests),
                )
                raw_items = result.output.items if result.output else []
                trace.refinement_applied = True
                trace.raw_llm_output = str(raw_items)

        items = _clean_items(raw_items, plan=plan, meta=meta)
        trace.cleaned_item_ids = [i.item_id for i in items]
        trace.diversity_applied = len(items) > 1
        trace.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        trace.usage = usage_summary(result)

        out = dataclasses.replace(result, output=RankedItems(items=items))
        out.trace = trace  # type: ignore[attr-defined]
        return out

    def run(self, request: str, deps: ToolRegistry) -> Any:
        """Sync convenience wrapper for the CLI.

        Handles the case where an event loop is already running (e.g. Jupyter,
        asyncio nested calls) by delegating to a thread.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(request, deps))
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.arun(request, deps)).result()
