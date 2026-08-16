"""Explainable recommendations: why an item for a user, grounded in evidence.

The core question is *accountability*: a recommendation should come with the
facts that produced it. :func:`explain_recommendation` derives those facts
deterministically from the trained artefacts — the user's taste (their
highest-rated genres and titles), the engine's score for the item, and
item-item similarity against titles the user already rated. No LLM is involved
in the evidence itself; the LLM (see :class:`RecExplainer`) may later restate
the same facts as fluent prose, and that prose is held to the facts by
construction.

``Explanation`` is the machine-readable contract; ``snippet``/``render()`` is
a short deterministic sentence for UIs that don't want to call the LLM.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from recagent.config import LLMConfig
from recagent.tools import ItemEntry, ToolRegistry

_COLD_START = "popularity"


class ContrastComparison(BaseModel):
    """Why this item was chosen over the best alternative."""

    alt_item_id: int
    alt_title: str
    alt_genres: list[str] = Field(default_factory=list)
    alt_score: float | None = None
    reason: str = ""


class Explanation(BaseModel):
    """The verifiable evidence behind one (user, item) recommendation."""

    item_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    score: float | None = None
    user_mean: float | None = None
    boost: float | None = None
    user_top_genres: list[str] = Field(default_factory=list)
    matched_genres: list[str] = Field(default_factory=list)
    user_likes: list[ItemEntry] = Field(default_factory=list)
    similar_rated: list[ItemEntry] = Field(default_factory=list)
    avg_rating: float | None = None
    rating_count: int = 0
    basis: str = ""
    snippet: str = ""
    contrast: ContrastComparison | None = None

    def render(self) -> str:
        """Deterministic one-sentence explanation — the no-LLM fallback."""
        return self.snippet


class ExplanationText(BaseModel):
    """The LLM's fluent restatement of an evidence block."""

    text: str = Field(description="one or two short sentences, only evidence facts")


def _user_ratings(deps: ToolRegistry, user_id: int) -> tuple[int, Any] | None:
    """(user_idx, csr row) for a known user, else None."""
    user_idx = deps.uid_to_idx.get(user_id)
    if user_idx is None:
        return None
    return user_idx, deps.matrix.getrow(user_idx)


def user_top_genres(
    deps: ToolRegistry,
    user_id: int,
    k: int = 6,
    *,
    _row: tuple[int, Any] | None = None,
) -> list[str]:
    """The user's dominant genres, weighted by rating points, best first.

    Every rated item contributes its rating value to each of its genres, so
    the ordering reflects intensity of preference, not just counts.
    """
    row = _row or _user_ratings(deps, user_id)
    if row is None:
        return []
    _, ratings_row = row
    points: dict[str, float] = {}
    for item_idx, rating in zip(ratings_row.indices, ratings_row.data):
        for genre in deps.items_meta.get(deps.item_ids[item_idx], {}).get("genres", []):
            points[genre] = points.get(genre, 0.0) + float(rating)
    ordered = sorted(points.items(), key=lambda kv: (-kv[1], kv[0]))
    return [genre for genre, _ in ordered[:k]]


def _engine_score(deps: ToolRegistry, user_id: int, item_id: int) -> float | None:
    """The engine's score for an item if it surfaces in the top-50, else None."""
    for entry in deps.recommend(user_id, n=50).items:
        if entry.item_id == item_id:
            return round(float(entry.score or 0.0), 4)
    return None


def _user_likes(
    deps: ToolRegistry,
    user_id: int,
    shared: set[str],
    k: int = 3,
) -> list[ItemEntry]:
    """The user's highest-rated titles, shared-genre ones first."""
    profile = deps.user_profile(user_id, k=10)
    shared_entries = [e for e in profile.items if shared & set(e.genres)]
    rest = [e for e in profile.items if not (shared & set(e.genres))]
    return (shared_entries + rest)[:k]


def _similar_rated(
    deps: ToolRegistry,
    user_id: int,
    item_id: int,
    k: int = 3,
) -> list[ItemEntry]:
    """Titles the user rated that are item-item similar to the target."""
    row = _user_ratings(deps, user_id)
    if row is None:
        return []
    _, ratings_row = row
    rated_ids = {deps.item_ids[idx] for idx in ratings_row.indices}
    out: list[ItemEntry] = []
    for entry in deps.similar_items(item_id, n=20).items:
        if entry.item_id in rated_ids:
            out.append(entry)
        if len(out) == k:
            break
    return out


def _find_contrast(
    deps: ToolRegistry,
    user_id: int,
    item_id: int,
    recommended_ids: set[int],
    *,
    score_a: float | None,
    score_b: float | None,
) -> ContrastComparison | None:
    """Find the best alternative the user likes and explain why this item was preferred.

    The alternative is the user's highest-rated item that shares at least one
    genre with the target but was *not* recommended.  If no such item exists,
    returns None.
    """
    info_a = deps.items_meta.get(item_id, {})
    genres_a = set(info_a.get("genres", []))
    profile = deps.user_profile(user_id, k=10)
    # find the best alternative: shares a genre with A, is rated highly, not recommended
    candidates: list[tuple[float, ItemEntry]] = []
    for entry in profile.items:
        if entry.item_id in recommended_ids or entry.item_id == item_id:
            continue
        genres_b = set(entry.genres)
        if not (genres_a & genres_b):
            continue
        candidates.append((entry.rating or 0.0, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    _, best = candidates[0]
    # build the reason string
    shared = genres_a & set(best.genres)
    parts: list[str] = []
    if score_a is not None and score_b is not None:
        diff = score_a - score_b
        if diff > 0:
            parts.append(f"engine score {score_a:.3f} vs {best.title}'s {score_b:.3f}")
    if shared:
        parts.append(f"both share {', '.join(sorted(shared))}")
    if best.rating:
        parts.append(f"you rated {best.title} {best.rating}/5")
    reason = f"Chosen over {best.title}: {'; '.join(parts)}." if parts else f"Chosen over {best.title}."
    return ContrastComparison(
        alt_item_id=best.item_id,
        alt_title=best.title,
        alt_genres=list(best.genres),
        alt_score=score_b,
        reason=reason,
    )


def _snippet(explanation: Explanation) -> str:
    """A deterministic sentence that only repeats facts already computed."""
    likes = explanation.user_likes
    similar = explanation.similar_rated
    if explanation.basis == _COLD_START:
        if explanation.rating_count:
            rating = f"{explanation.avg_rating:.1f}" if explanation.avg_rating else "?"
            return (
                f"{explanation.title} is a crowd favourite: "
                f"rated {explanation.rating_count} times at {rating}/5."
            )
        return f"{explanation.title} — no ratings yet."
    if similar:
        first = similar[0]
        base = (
            f"You rated {first.title}, which is similar to {explanation.title} "
            f"(similarity {first.score:.2f})."
        )
    elif explanation.matched_genres:
        genre = explanation.matched_genres[0]
        like = likes[0] if likes else None
        prefix = (
            f"Alongside your {like.rating}/5 for {like.title}, " if like and like.rating else ""
        )
        base = f"{prefix}{explanation.title} fits your {genre} taste."
    elif likes:
        like = likes[0]
        base = (
            f"You rated {like.title} {like.rating}/5 — "
            f"{explanation.title} sits in your wider taste."
        )
    elif explanation.rating_count:
        base = f"{explanation.title} is popular: {explanation.rating_count} ratings."
    else:
        base = f"{explanation.title}."
    if explanation.contrast:
        base += f" {explanation.contrast.reason}"
    return base


def explain_recommendation(
    deps: ToolRegistry,
    user_id: int,
    item_id: int,
    *,
    k: int = 3,
    recommended_ids: set[int] | None = None,
) -> Explanation:
    """Deterministic, verifiable evidence for recommending ``item_id`` to ``user_id``."""
    info = deps.item_info(item_id)
    genres = list(info.genres)

    row = _user_ratings(deps, user_id)
    if row is None:
        cold = Explanation(
            item_id=item_id,
            title=info.title,
            genres=genres,
            avg_rating=info.avg_rating,
            rating_count=info.rating_count,
            basis=_COLD_START,
        )
        cold.snippet = _snippet(cold)
        return cold

    _, ratings_row = row
    user_mean = round(float(ratings_row.data.mean()), 4) if ratings_row.data.size else None
    top_genres = user_top_genres(deps, user_id, _row=row)
    shared = set(genres) & set(top_genres)
    likes = _user_likes(deps, user_id, shared, k=k)
    similar = _similar_rated(deps, user_id, item_id, k=k)

    if shared:
        basis = "genre-affinity"
    elif similar:
        basis = "similar-taste"
    elif likes:
        basis = "taste-overlap"
    else:
        basis = _COLD_START

    score = _engine_score(deps, user_id, item_id)
    # contrastive comparison: why this item over the best alternative?
    contrast = None
    if recommended_ids:
        # find the alternative's engine score from the CF candidates
        alt_score = None
        for entry in deps.recommend(user_id, n=50).items:
            if entry.item_id != item_id:
                alt_score = float(entry.score or 0.0)
                break
        contrast = _find_contrast(
            deps, user_id, item_id, recommended_ids, score_a=score, score_b=alt_score
        )

    explanation = Explanation(
        item_id=item_id,
        title=info.title,
        genres=genres,
        score=score,
        user_mean=user_mean,
        boost=round(score - user_mean, 4) if score is not None and user_mean is not None else None,
        user_top_genres=top_genres,
        matched_genres=[g for g in genres if g in shared],
        user_likes=likes,
        similar_rated=similar,
        avg_rating=info.avg_rating,
        rating_count=info.rating_count,
        basis=basis,
        contrast=contrast,
    )
    explanation.snippet = _snippet(explanation)
    return explanation


def _evidence_block(explanation: Explanation) -> str:
    """Render the evidence as text for the LLM (facts only, no interpretation)."""
    lines = [
        (
            f"item: {explanation.item_id} — {explanation.title} "
            f"[{', '.join(explanation.genres) or '-'}]"
        ),
        f"user mean rating: {explanation.user_mean}",
        f"engine score: {explanation.score}",
        f"user's top genres: {', '.join(explanation.user_top_genres) or 'none'}",
        f"matched genres: {', '.join(explanation.matched_genres) or 'none'}",
    ]
    if explanation.user_likes:
        lines.append(
            "user's liked titles: "
            + "; ".join(
                f"{e.title} ({e.rating}/5)" for e in explanation.user_likes if e.rating
            )
        )
    if explanation.similar_rated:
        lines.append(
            "similar titles user rated: "
            + "; ".join(
                f"{e.title} (similarity {e.score:.2f})"
                for e in explanation.similar_rated
                if e.score is not None
            )
        )
    if explanation.rating_count:
        lines.append(
            f"catalog: rated {explanation.rating_count}x, avg {explanation.avg_rating}/5"
        )
    if explanation.contrast:
        c = explanation.contrast
        lines.append(
            f"contrastive: preferred over {c.alt_title} ({', '.join(c.alt_genres) or '-'}), "
            f"{c.reason}"
        )
    return "\n".join(lines)


_EXPLAIN_PROMPT = """\
You explain a single recommendation to the user. Below is the only evidence
you may use: the item, the user's profile facts, and the engine score.

Rules:
- Restate ONLY facts present in the evidence. Never invent titles, genres,
  ratings, or scores.
- One or two short, friendly sentences. Start by addressing the user as "you".
- Do not mention user ids, engine scores, or the word "evidence".
"""


class RecExplainer:
    """Turns a deterministic :class:`Explanation` into fluent, grounded prose.

    A single structured-output call (no tools). When the LLM is disabled the
    caller falls back to ``explanation.snippet``, so explanations always exist.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        temperature: float = 0.1,
        agent: Any | None = None,
    ):
        self.config = config
        if agent is not None:
            self.agent = agent
            return
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.settings import ModelSettings

        model = OpenAIChatModel(
            config.model,
            provider=OpenAIProvider(base_url=config.base_url, api_key=config.api_key),
        )
        self.agent = Agent(
            model,
            name="recagent-explain",
            output_type=ExplanationText,
            system_prompt=_EXPLAIN_PROMPT,
            retries=2,
            defer_model_check=True,
            model_settings=ModelSettings(temperature=temperature, max_tokens=200),
        )

    async def aexplain(self, explanation: Explanation) -> tuple[str, dict[str, int]]:
        """Return (grounded text, usage) for a pre-computed evidence block."""
        from pydantic_ai.usage import UsageLimits

        result = await self.agent.run(
            _evidence_block(explanation),
            usage_limits=UsageLimits(request_limit=4),
        )
        output = result.output
        text = output.text.strip() if output else ""
        if not text:
            text = explanation.snippet  # guardrail: never return an empty line
        return text, usage_summary(result)

    def explain(self, explanation: Explanation) -> tuple[str, dict[str, int]]:
        """Sync convenience wrapper for the CLI."""
        import asyncio

        return asyncio.run(self.aexplain(explanation))


def usage_summary(result: Any) -> dict[str, int]:
    usage = result.usage() if callable(getattr(result, "usage", None)) else result.usage
    return {
        "requests": getattr(usage, "requests", 0),
        "prompt_tokens": getattr(usage, "input_tokens", getattr(usage, "request_tokens", 0)),
        "completion_tokens": getattr(
            usage, "output_tokens", getattr(usage, "response_tokens", 0)
        ),
    }
