import asyncio

import httpx

from recagent_sdk import RecagentError, RecommendClient


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/health":
        return httpx.Response(
            200,
            json={"status": "ok", "agent_enabled": True, "model": "x", "n_users": 943, "n_items": 1682},
        )
    if request.url.path == "/recommend":
        return httpx.Response(
            200,
            json={
                "user_id": 1,
                "k": 2,
                "items": [
                    {"item_id": 10, "title": "Alpha", "genres": ["Sci-Fi"], "score": 0.9, "reason": "top"}
                ],
                "usage": {"requests": 1, "prompt_tokens": 10, "completion_tokens": 5},
            },
        )
    if request.url.path == "/chat":
        return httpx.Response(
            200, json={"user_id": 1, "items": [], "evidence": "evidence", "usage": {}}
        )
    if request.url.path == "/feedback":
        return httpx.Response(
            200,
            json={"accepted": True, "event": {"user_id": 1, "item_id": 10, "liked": True}},
        )
    if request.url.path == "/explain":
        return httpx.Response(
            200,
            json={
                "user_id": 1,
                "explanation": {
                    "item_id": 10,
                    "title": "Alpha",
                    "genres": ["Sci-Fi"],
                    "basis": "genre-affinity",
                    "matched_genres": ["Sci-Fi"],
                    "snippet": "fits your Sci-Fi taste",
                },
                "text": "You love sci-fi, and Alpha is the pick.",
                "llm": True,
                "usage": {"requests": 1},
            },
        )
    if request.url.path == "/catalog/10":
        return httpx.Response(
            200,
            json={"item_id": 10, "title": "Alpha", "genres": ["Sci-Fi"], "rating_count": 5, "avg_rating": 4.2},
        )
    return httpx.Response(500, text="boom")


def _client() -> RecommendClient:
    return RecommendClient(transport=httpx.MockTransport(_handler))


def test_health():
    with _client() as c:
        h = c.health()
        assert h.n_users == 943
        assert h.agent_enabled is True


def test_recommend_typed():
    with _client() as c:
        resp = c.recommend(1, k=2, filters={"genre": "Sci-Fi"})
        assert resp.user_id == 1
        assert resp.items[0].title == "Alpha"
        assert resp.items[0].score == 0.9
        assert resp.items[0].reason == "top"
        assert resp.usage["requests"] == 1


def test_chat_and_feedback_and_catalog():
    with _client() as c:
        chat = c.chat("sci-fi", user_id=1)
        assert chat.evidence == "evidence"
        fb = c.feedback(1, 10, liked=True)
        assert fb.accepted is True
        entry = c.catalog(10)
        assert entry.avg_rating == 4.2
        assert entry.genres == ["Sci-Fi"]


def test_explain_typed():
    with _client() as c:
        resp = c.explain(1, 10)
        assert resp.llm is True
        assert resp.text == "You love sci-fi, and Alpha is the pick."
        assert resp.explanation.basis == "genre-affinity"
        assert resp.explanation.matched_genres == ["Sci-Fi"]
        assert resp.usage["requests"] == 1


def test_error_raises():
    with _client() as c:
        try:
            c.catalog(999)  # unmocked path -> 500
            raise AssertionError("expected RecagentError")
        except RecagentError as exc:
            assert "boom" in str(exc)


def test_async_client():
    async def run():
        async with _client() as client:
            resp = await client.arecommend(1, k=2)
            return resp.items[0].score

    assert asyncio.run(run()) == 0.9
