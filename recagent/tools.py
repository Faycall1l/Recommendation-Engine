"""Tool registry: evidence-grounded access to the collaborative filtering engine.

Each public method maps 1:1 to a pydantic-ai tool. Signatures (and their
docstrings) are turned into JSON schemas by pydantic-ai automatically.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from recagent.state import load_state


class ItemEntry(BaseModel):
    item_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    rating_count: int = 0
    avg_rating: float | None = None
    score: float | None = None
    rating: float | None = None


class ItemList(BaseModel):
    items: list[ItemEntry]
    user_id: int | None = None
    query: str | None = None


class ToolRegistry:
    """Binds the tools to a trained model state; pydantic-ai injects this as deps."""

    def __init__(self, state: dict[str, Any]):
        self.state = state
        self.model = state["model"]
        self.matrix = state["matrix"]
        self.uid_to_idx = state["uid_to_idx"]
        self.iid_to_idx = state["iid_to_idx"]
        self.user_ids = state["user_ids"]
        self.item_ids = state["item_ids"]
        self.items_meta = state["items_meta"]

    @classmethod
    def from_artifacts(cls, artifacts_dir: str = "artifacts") -> ToolRegistry:
        return cls(load_state(artifacts_dir))

    def _item_meta(self, item_id: int) -> ItemEntry:
        info = self.items_meta.get(item_id, {})
        idx = self.iid_to_idx[item_id]
        col = self.matrix.getcol(idx)
        count = int(col.nnz)
        avg = float(col.sum()) / count if count else None
        return ItemEntry(
            item_id=int(item_id),
            title=info.get("title", f"<item {item_id}>"),
            genres=info.get("genres", []),
            rating_count=count,
            avg_rating=round(avg, 2) if avg is not None else None,
        )

    def recommend(self, user_id: int, n: int = 10) -> ItemList:
        """Top-n candidate items for a user from the CF engine, best first.

        Already-seen items are excluded by the engine.
        """
        user_idx = self.uid_to_idx[user_id]
        items = [
            self._item_meta(self.item_ids[item_idx]).model_copy(
                update={"score": round(float(score), 4)}
            )
            for item_idx, score in self.model.recommend(self.matrix, user_idx, n=n)
        ]
        return ItemList(user_id=user_id, items=items)

    def user_profile(self, user_id: int, k: int = 8) -> ItemList:
        """The user's highest-rated items — their taste profile."""
        user_idx = self.uid_to_idx[user_id]
        row = self.matrix.getrow(user_idx)
        order = row.data.argsort()[::-1][:k]
        items = []
        for pos in order:
            item_idx = row.indices[pos]
            item_id = self.item_ids[item_idx]
            items.append(
                self._item_meta(item_id).model_copy(
                    update={"rating": round(float(row.data[pos]), 1)}
                )
            )
        return ItemList(user_id=user_id, items=items)

    def item_info(self, item_id: int) -> ItemEntry:
        """Metadata for a single item: title, genres, popularity."""
        return self._item_meta(item_id)

    def search_items(self, query: str, n: int = 10) -> ItemList:
        """Items whose title or genres match a free-text query."""
        tokens = {t for t in query.lower().split() if t}
        scored: list[tuple[int, int, int]] = []
        for item_id, idx in self.iid_to_idx.items():
            info = self.items_meta.get(item_id, {})
            haystack = (info.get("title", "") + " " + " ".join(info.get("genres", []))).lower()
            matches = sum(1 for t in tokens if t in haystack)
            if matches:
                scored.append((matches, -idx, item_id))
        scored.sort(reverse=True)
        return ItemList(
            query=query,
            items=[self._item_meta(item_id) for _, _, item_id in scored[:n]],
        )
