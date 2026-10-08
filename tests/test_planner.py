import json
import httpx
from finrag.planner import ModelPlanner
from finrag.core import Retriever, chunk_blocks
from finrag.finqa import blocks_from_documents


def test_model_plan_rejects_unretrieved_cells_and_records_raw_output(monkeypatch):
    docs = [{"doc_id": "a", "company": "ACME", "report_year": "2024", "page": 1,
             "pre_text": [], "post_text": [], "table": [["metric", "2024", "2023"], ["Revenue", "120", "100"]]}]
    r = Retriever(chunk_blocks(blocks_from_documents(docs)))
    plan = {"operation": "growth", "operands": [{"doc_id": "a", "row": 1, "column": 1}, {"doc_id": "a", "row": 1, "column": 2}]}
    original = httpx.Client
    def respond(request):
        body = json.loads(request.content)
        assert "gold_inds" not in json.dumps(body) and "exe_ans" not in json.dumps(body)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(plan)}}]})
    monkeypatch.setattr(httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(respond)))
    planner = ModelPlanner("http://fixture/v1", "fixture")
    answer = planner.answer(r, docs, "revenue growth 2023 2024")
    assert not answer["refused"] and answer["value"] == "20.0" and answer["model_trace"]["response"]
    plan["operands"][0]["doc_id"] = "unretrieved"
    assert planner.answer(r, docs, "revenue growth 2023 2024")["refused"]
    plan["operands"][0]["doc_id"] = "a"
    plan["operands"][0]["column"] = 999
    assert planner.answer(r, docs, "revenue growth 2023 2024")["refused"]
