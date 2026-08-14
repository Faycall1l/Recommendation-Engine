from fastapi.testclient import TestClient

from recagent.api import create_app


class StubClient:
    async def arecommend(self, user_id, k=5, filters=None):
        return {"user_id": user_id, "k": k, "items": [{"item_id": 10, "title": "Alpha"}], "usage": {}}

    async def achat(self, message, user_id=None):
        return {"user_id": user_id, "items": [{"item_id": 10, "title": "Alpha"}], "evidence": "e", "usage": {}}

    async def aexplain_recommendation(self, user_id, item_id):
        return {
            "user_id": user_id,
            "explanation": {"item_id": item_id, "title": "Alpha", "genres": ["Sci-Fi"], "basis": "popularity", "snippet": "s"},
            "text": "grounded prose",
            "llm": True,
            "usage": {"requests": 1},
        }

    def feedback(self, user_id, item_id, liked):
        return {"accepted": True, "event": {"user_id": user_id, "item_id": item_id, "liked": liked}}

    def explain(self, item_id):
        return {"item_id": item_id, "title": "Alpha", "genres": ["Sci-Fi"]}

    def health(self):
        return {"status": "ok", "agent_enabled": True, "n_users": 943, "n_items": 1682}


def _client() -> TestClient:
    return TestClient(create_app(client=StubClient()))


def test_health():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["n_users"] == 943


def test_recommend():
    resp = _client().post("/recommend", json={"user_id": 1, "k": 5, "filters": {"genre": "Sci-Fi"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == 1
    assert body["items"][0]["item_id"] == 10


def test_recommend_rejects_bad_k():
    resp = _client().post("/recommend", json={"user_id": 1, "k": 0})
    assert resp.status_code == 422


def test_chat():
    resp = _client().post("/chat", json={"message": "sci-fi", "user_id": 1})
    assert resp.status_code == 200
    assert resp.json()["items"][0]["title"] == "Alpha"


def test_explain():
    resp = _client().post("/explain", json={"user_id": 1, "item_id": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "grounded prose"
    assert body["llm"] is True
    assert body["explanation"]["basis"] == "popularity"


def test_feedback():
    resp = _client().post("/feedback", json={"user_id": 1, "item_id": 10, "liked": True})
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True


def test_catalog():
    resp = _client().get("/catalog/10")
    assert resp.status_code == 200
    assert resp.json()["title"] == "Alpha"
