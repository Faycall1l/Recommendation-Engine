"""User memory: structured preference buckets backed by JSON.

Preferences are stored per-user in named buckets (e.g. ``loved``,
``disliked``, ``comfort``, ``discovery``, ``mood:relaxed``).  Each bucket
holds a list of item IDs with timestamps, enabling the agent to reason
about a user's evolving taste rather than raw CF scores alone.

Storage is a single JSON file; reads are in-memory after first load.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field


class PreferenceEntry(BaseModel):
    """A single preference record inside a bucket."""

    item_id: int
    added_at: float = Field(default_factory=time.time)
    source: str = "feedback"  # "feedback" | "explicit" | "inferred"
    note: str = ""


class PreferenceBucket(BaseModel):
    """Named collection of preference entries for one user."""

    category: str
    entries: list[PreferenceEntry] = Field(default_factory=list)

    def item_ids(self) -> list[int]:
        return [e.item_id for e in self.entries]

    def has(self, item_id: int) -> bool:
        return any(e.item_id == item_id for e in self.entries)


class UserMemory:
    """Per-user preference storage with bucket semantics.

    Buckets are arbitrary strings; common conventions:

    - ``loved`` — items the user explicitly liked or rated highly
    - ``disliked`` — items the user disliked or rated poorly
    - ``comfort`` — reliable favourites, comfort rewatch material
    - ``discovery`` — new or surprising items that worked
    - ``mood:<name>`` — mood-tagged preferences (e.g. ``mood:relaxed``)
    - ``context:<name>`` — context-tagged (e.g. ``context:late_night``)
    - ``genre:<name>`` — genre-specific taste signals

    The ``memory.json`` file is a flat dict keyed by ``str(user_id)``.
    """

    def __init__(self, path: str | Path = "artifacts/memory.json"):
        self._path = Path(path)
        self._data: dict[int, dict[str, PreferenceBucket]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._path.exists():
            raw = json.loads(self._path.read_text())
            for uid_str, buckets in raw.items():
                uid = int(uid_str)
                self._data[uid] = {}
                for cat, entries in buckets.items():
                    self._data[uid][cat] = PreferenceBucket(
                        category=cat,
                        entries=[PreferenceEntry(**e) for e in entries],
                    )
        self._loaded = True

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out: dict[str, dict[str, list[dict]]] = {}
        for uid, buckets in self._data.items():
            out[str(uid)] = {
                cat: [e.model_dump() for e in bucket.entries]
                for cat, bucket in buckets.items()
            }
        self._path.write_text(json.dumps(out, indent=2))

    def save_preference(
        self,
        user_id: int,
        category: str,
        item_ids: list[int],
        *,
        source: str = "explicit",
        note: str = "",
    ) -> int:
        """Add item_ids to a bucket. Returns the number of new entries added."""
        self._ensure_loaded()
        if user_id not in self._data:
            self._data[user_id] = {}
        bucket = self._data[user_id].setdefault(
            category, PreferenceBucket(category=category)
        )
        existing = set(bucket.item_ids())
        added = 0
        for iid in item_ids:
            if iid not in existing:
                bucket.entries.append(
                    PreferenceEntry(item_id=iid, source=source, note=note)
                )
                added += 1
        if added:
            self._save()
        return added

    def get_preferences(
        self, user_id: int, category: str | None = None
    ) -> dict[str, list[int]]:
        """Return item_ids per bucket, optionally filtered to one category."""
        self._ensure_loaded()
        buckets = self._data.get(user_id, {})
        if category:
            b = buckets.get(category)
            return {category: b.item_ids()} if b else {}
        return {cat: b.item_ids() for cat, b in buckets.items()}

    def get_preference_summary(self, user_id: int) -> str:
        """Human-readable summary of all preference buckets for evidence."""
        self._ensure_loaded()
        buckets = self._data.get(user_id, {})
        if not buckets:
            return ""
        lines: list[str] = []
        for cat, bucket in sorted(buckets.items()):
            ids = bucket.item_ids()
            if ids:
                lines.append(f"  {cat}: {len(ids)} items (IDs: {ids[:10]})")
        return "\n".join(lines)

    def remove_preference(
        self, user_id: int, category: str, item_id: int
    ) -> bool:
        """Remove an item from a bucket. Returns True if removed."""
        self._ensure_loaded()
        bucket = self._data.get(user_id, {}).get(category)
        if not bucket:
            return False
        before = len(bucket.entries)
        bucket.entries = [e for e in bucket.entries if e.item_id != item_id]
        removed = len(bucket.entries) < before
        if removed:
            self._save()
        return removed

    def list_users(self) -> list[int]:
        """All user IDs that have any preferences stored."""
        self._ensure_loaded()
        return sorted(self._data.keys())

    def list_categories(self, user_id: int) -> list[str]:
        """All bucket names for a user."""
        self._ensure_loaded()
        return sorted(self._data.get(user_id, {}).keys())

    def ingest_feedback(
        self, feedback_path: str | Path, *, min_rating: float = 3.5
    ) -> int:
        """Read a feedback JSONL file and populate love/dislike buckets.

        Items with ``liked=True`` go to ``loved``; items with ``liked=False``
        go to ``disliked``.  Returns the number of new entries added.
        """
        self._ensure_loaded()
        fb_path = Path(feedback_path)
        if not fb_path.exists():
            return 0
        added = 0
        for line in fb_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = event.get("user_id")
            iid = event.get("item_id")
            liked = event.get("liked")
            if uid is None or iid is None or liked is None:
                continue
            cat = "loved" if liked else "disliked"
            added += self.save_preference(
                int(uid), cat, [int(iid)], source="feedback"
            )
        return added

    def clear(self, user_id: int, category: str | None = None) -> None:
        """Clear all buckets for a user, or one specific bucket."""
        self._ensure_loaded()
        if user_id not in self._data:
            return
        if category:
            self._data[user_id].pop(category, None)
        else:
            del self._data[user_id]
        self._save()
