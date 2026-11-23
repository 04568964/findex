from fastapi.testclient import TestClient

from findex.models import DocMeta, Index, Posting
from findex.web.app import app, get_index, get_index_optional


def _make_fake_index() -> Index:
    docs = {
        0: DocMeta(doc_id=0, path="doc0.txt", title="Python Async", length=3),
        1: DocMeta(doc_id=1, path="doc1.txt", title="FastAPI Web", length=3),
    }
    postings = {
        "python": [Posting(doc_id=0, freq=2, positions=(0, 2))],
        "async": [Posting(doc_id=0, freq=1, positions=(1,))],
        "fastapi": [Posting(doc_id=1, freq=1, positions=(0,))],
    }
    return Index(postings, docs, has_positions=True)


def test_health_unavailable_when_index_missing() -> None:
    app.dependency_overrides[get_index_optional] = lambda: None
    try:
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json() == {"status": "unavailable", "index_loaded": False}
    finally:
        app.dependency_overrides.clear()


def test_search_stats_and_doc_with_override() -> None:
    fake_idx = _make_fake_index()
    app.dependency_overrides[get_index_optional] = lambda: fake_idx
    app.dependency_overrides[get_index] = lambda: fake_idx
    try:
        client = TestClient(app)

        assert client.get("/health").status_code == 200

        st = client.get("/stats")
        assert st.status_code == 200
        assert st.json()["documents"] == 2

        res = client.get("/search", params={"q": "python", "k": 5, "page": 1})
        assert res.status_code == 200
        body = res.json()
        assert body["total"] == 1
        assert body["results"][0]["doc_id"] == 0

        doc = client.get("/docs/0")
        assert doc.status_code == 200
        assert doc.json()["title"] == "Python Async"

        missing = client.get("/docs/999")
        assert missing.status_code == 404

        invalid = client.get("/search", params={"q": "", "k": 0})
        assert invalid.status_code == 422
    finally:
        app.dependency_overrides.clear()