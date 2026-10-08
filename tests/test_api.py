import json
from fastapi.testclient import TestClient
from finrag import api


def test_custom_pdf_blocks_can_be_queried_via_api(tmp_path, monkeypatch):
    path = tmp_path / "parsed.json"
    path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "own-report",
                    "page": 3,
                    "text": "Audited 2024 revenue is 100 million dollars.",
                    "kind": "text",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_DATA", str(path))
    monkeypatch.setattr(api, "retriever", None)
    monkeypatch.setattr(api, "docs", [])
    monkeypatch.setattr(api, "config", {})
    with TestClient(api.app) as client:
        assert client.get("/health").json()["documents"] == 1
        result = client.post(
            "/query",
            json={"query": "What is the revenue in 2024?", "doc_id": "own-report"},
        ).json()
        assert not result["refused"] and result["citations"][0]["page"] == 3
