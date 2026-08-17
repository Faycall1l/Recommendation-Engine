import json

from recagent.memory import UserMemory
from recagent.session import SessionMemory
from recagent.tools import ToolRegistry


def _make_state(n_users=10, n_items=20):
    import numpy as np
    import scipy.sparse as sp

    from recagent.model import Recommender

    rng = np.random.default_rng(0)
    rows, cols, data = [], [], []
    for u in range(n_users):
        for _ in range(int(rng.integers(4, 8))):
            rows.append(u)
            cols.append(int(rng.integers(0, n_items)))
            data.append(float(rng.integers(1, 6)))
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    user_ids = np.arange(101, 101 + n_users)
    item_ids = np.arange(1, 1 + n_items)
    model = Recommender(factors=8, iterations=5).fit(matrix)
    items_meta = {int(i): {"title": f"Item {i}", "genres": ["Drama"]} for i in item_ids}
    return {
        "model": model,
        "matrix": matrix,
        "uid_to_idx": {int(u): i for i, u in enumerate(user_ids)},
        "iid_to_idx": {int(i): j for j, i in enumerate(item_ids)},
        "user_ids": user_ids,
        "item_ids": item_ids,
        "items_meta": items_meta,
    }


class TestUserMemory:
    def test_add_and_get(self, tmp_path):
        mem = UserMemory(tmp_path / "mem.json")
        mem.save_preference(42, "loved", [10], source="explicit")
        mem.save_preference(42, "loved", [11], source="explicit")
        mem.save_preference(42, "hated", [5], source="explicit")
        buckets = mem.get_preferences(42)
        assert len(buckets["loved"]) == 2
        assert len(buckets["hated"]) == 1
        assert 10 in buckets["loved"]

    def test_persistence(self, tmp_path):
        path = tmp_path / "mem.json"
        mem1 = UserMemory(path)
        mem1.save_preference(1, "loved", [100])
        mem2 = UserMemory(path)
        buckets = mem2.get_preferences(1)
        assert len(buckets["loved"]) == 1

    def test_summary(self, tmp_path):
        mem = UserMemory(tmp_path / "mem.json")
        mem.save_preference(7, "loved", [501, 502])
        mem.save_preference(7, "hated", [503])
        summary = mem.get_preference_summary(7)
        assert "loved" in summary
        assert "501" in summary

    def test_empty_user_summary(self, tmp_path):
        mem = UserMemory(tmp_path / "mem.json")
        assert mem.get_preference_summary(999) == ""

    def test_ingest_feedback(self, tmp_path):
        feedback = tmp_path / "fb.jsonl"
        feedback.write_text(
            json.dumps({"user_id": 5, "item_id": 101, "liked": True})
            + "\n"
            + json.dumps({"user_id": 5, "item_id": 102, "liked": False})
            + "\n"
            + json.dumps({"user_id": 5, "item_id": 103, "liked": True})
            + "\n"
        )
        mem = UserMemory(tmp_path / "mem.json")
        mem.ingest_feedback(feedback)
        buckets = mem.get_preferences(5)
        assert len(buckets["loved"]) == 2
        assert len(buckets["disliked"]) == 1

    def test_dedup(self, tmp_path):
        mem = UserMemory(tmp_path / "mem.json")
        n = mem.save_preference(1, "loved", [10, 20])
        assert n == 2
        n2 = mem.save_preference(1, "loved", [10, 30])
        assert n2 == 1  # only 30 is new
        assert mem.get_preferences(1)["loved"] == [10, 20, 30]

    def test_remove(self, tmp_path):
        mem = UserMemory(tmp_path / "mem.json")
        mem.save_preference(1, "loved", [10, 20])
        assert mem.remove_preference(1, "loved", 10)
        assert mem.get_preferences(1)["loved"] == [20]

    def test_list_users_and_categories(self, tmp_path):
        mem = UserMemory(tmp_path / "mem.json")
        mem.save_preference(1, "loved", [10])
        mem.save_preference(1, "hated", [20])
        mem.save_preference(2, "loved", [30])
        assert mem.list_users() == [1, 2]
        assert mem.list_categories(1) == ["hated", "loved"]

    def test_clear(self, tmp_path):
        mem = UserMemory(tmp_path / "mem.json")
        mem.save_preference(1, "loved", [10])
        mem.clear(1)
        assert mem.get_preferences(1) == {}


class TestSessionMemory:
    def test_record_recommendation(self):
        s = SessionMemory()
        s.record_recommendation("find me sci-fi", [1, 2, 3])
        assert s.turn_count == 1
        assert s.recently_recommended() == [1, 2, 3]

    def test_record_feedback(self):
        s = SessionMemory()
        s.record_recommendation("horror movies", [10, 20])
        s.record_feedback(10, liked=True)
        s.record_feedback(20, liked=False)
        assert s.liked_this_session() == [10]
        assert s.disliked_this_session() == [20]

    def test_max_turns(self):
        s = SessionMemory(max_turns=3)
        for i in range(5):
            s.record_recommendation(f"req {i}", [i])
        assert s.turn_count == 3
        assert s.recently_recommended() == [4, 3, 2]

    def test_session_summary(self):
        s = SessionMemory()
        s.record_recommendation("find rom-coms", [5, 6])
        s.record_feedback(5, liked=True)
        summary = s.session_summary()
        assert "rom-coms" in summary
        assert "liked" in summary

    def test_empty_session_summary(self):
        s = SessionMemory()
        assert s.session_summary() == ""

    def test_clear(self):
        s = SessionMemory()
        s.record_recommendation("req", [1])
        s.clear()
        assert s.turn_count == 0

    def test_session_turn_independent(self):
        s = SessionMemory()
        s.record_recommendation("first", [1, 2])
        s.record_recommendation("second", [3, 4])
        assert s.recently_recommended() == [3, 4, 1, 2]


class TestToolRegistryMemory:
    def test_save_preference(self, tmp_path):
        state = _make_state()
        registry = ToolRegistry(state, memory=UserMemory(tmp_path / "mem.json"))
        result = registry.save_preference(101, "loved", [3], note="smoke test")
        assert result["added"] == 1
        prefs = registry.memory.get_preferences(101)
        assert 3 in prefs["loved"]

    def test_get_preferences(self, tmp_path):
        state = _make_state()
        registry = ToolRegistry(state, memory=UserMemory(tmp_path / "mem.json"))
        registry.memory.save_preference(101, "hated", [7])
        result = registry.get_preferences(101)
        assert 7 in result["hated"]

    def test_get_preference_summary(self, tmp_path):
        state = _make_state()
        registry = ToolRegistry(state, memory=UserMemory(tmp_path / "mem.json"))
        registry.memory.save_preference(101, "loved", [100])
        result = registry.get_preference_summary(101)
        assert "100" in result

    def test_ingest_feedback_tool(self, tmp_path):
        state = _make_state()
        registry = ToolRegistry(state, memory=UserMemory(tmp_path / "mem.json"))
        fb_path = tmp_path / "fb.jsonl"
        fb_path.write_text(
            json.dumps({"user_id": 101, "item_id": 99, "liked": True}) + "\n"
        )
        result = registry.ingest_feedback(str(fb_path))
        assert result["ingested"] == 1
        prefs = registry.memory.get_preferences(101)
        assert 99 in prefs["loved"]

    def test_get_preferences_empty(self):
        state = _make_state()
        registry = ToolRegistry(state)
        result = registry.get_preferences(999)
        assert result == {}

    def test_session_wired(self):
        state = _make_state()
        registry = ToolRegistry(state)
        assert registry.session is not None
        assert registry.session.turn_count == 0
