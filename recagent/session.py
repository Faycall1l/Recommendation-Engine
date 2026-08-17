"""Session memory: in-conversation context for multi-turn recommendations.

Tracks what was recommended, liked, and discussed within a session so the
agent can avoid repetition and build on prior turns.  Not persisted to disk —
lives only for the lifetime of a RecClient or conversation.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class SessionTurn(BaseModel):
    """One turn in the conversation."""

    timestamp: float = Field(default_factory=time.time)
    request: str = ""
    recommended_item_ids: list[int] = Field(default_factory=list)
    liked_item_ids: list[int] = Field(default_factory=list)
    disliked_item_ids: list[int] = Field(default_factory=list)
    tags_used: list[str] = Field(default_factory=list)


class SessionMemory:
    """Tracks the conversation within a single session.

    The agent uses this to:
    - Avoid re recommending items from the current session
    - Understand what the user liked/disliked in this conversation
    - Detect patterns (e.g. "you keep recommending dramas, try something else")
    - Build context for follow-up requests ("more like that")
    """

    def __init__(self, max_turns: int = 20):
        self._turns: list[SessionTurn] = []
        self._max_turns = max_turns

    def add_turn(self, turn: SessionTurn) -> None:
        self._turns.append(turn)
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns:]

    def record_recommendation(self, request: str, item_ids: list[int]) -> None:
        turn = SessionTurn(request=request, recommended_item_ids=item_ids)
        self.add_turn(turn)

    def record_feedback(self, item_id: int, liked: bool) -> None:
        if not self._turns:
            self.add_turn(SessionTurn())
        last = self._turns[-1]
        if liked:
            last.liked_item_ids.append(item_id)
        else:
            last.disliked_item_ids.append(item_id)

    def recently_recommended(self, n: int = 20) -> list[int]:
        """All item_ids recommended in this session, most recent first."""
        ids: list[int] = []
        for turn in reversed(self._turns):
            ids.extend(turn.recommended_item_ids)
        return ids[:n]

    def liked_this_session(self) -> list[int]:
        """Item_ids the user liked in this session."""
        ids: list[int] = []
        for turn in self._turns:
            ids.extend(turn.liked_item_ids)
        return ids

    def disliked_this_session(self) -> list[int]:
        """Item_ids the user disliked in this session."""
        ids: list[int] = []
        for turn in self._turns:
            ids.extend(turn.disliked_item_ids)
        return ids

    def session_summary(self) -> str:
        """Human-readable summary for evidence injection."""
        if not self._turns:
            return ""
        recent = self._turns[-5:]
        lines: list[str] = []
        for i, turn in enumerate(recent, 1):
            n_rec = len(turn.recommended_item_ids)
            n_like = len(turn.liked_item_ids)
            n_dislike = len(turn.disliked_item_ids)
            parts = [f"Turn {i}: \"{turn.request}\"" if turn.request else f"Turn {i}"]
            if n_rec:
                parts.append(f"{n_rec} items recommended")
            if n_like:
                parts.append(f"liked {turn.liked_item_ids}")
            if n_dislike:
                parts.append(f"disliked {turn.disliked_item_ids}")
            lines.append(", ".join(parts))
        all_recommended = self.recently_recommended()
        all_liked = self.liked_this_session()
        all_disliked = self.disliked_this_session()
        lines.append(
            f"Session totals: {len(all_recommended)} recommended, "
            f"{len(all_liked)} liked, {len(all_disliked)} disliked"
        )
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)
