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
    answer = planner.answer(r, docs, "ACME report year 2024 revenue growth from 2023 to 2024")
    assert not answer["refused"] and answer["value"] == "20.0" and answer["model_trace"]["response"]
    assert answer["scope"]["source"] == "question_entity_year"
    assert planner.answer(r, docs, "revenue growth from 2023 to 2024")["refused"]
    plan["operands"][0]["doc_id"] = "unretrieved"
    assert planner.answer(r, docs, "ACME report year 2024 revenue growth")["refused"]
    plan["operands"][0]["doc_id"] = "a"
    plan["operands"][0]["column"] = 999
    assert planner.answer(r, docs, "ACME report year 2024 revenue growth")["refused"]


def test_model_plan_scopes_before_retrieval_and_allows_caller_report(monkeypatch):
    docs = [{"doc_id": f"{company}/{year}/p", "company": company, "report_year": year, "page": 1,
             "pre_text": [], "post_text": [], "table": [["metric", year], ["Revenue", value]]}
            for company, year, value in [("ACME", "2024", "100"), ("ACME", "2023", "90"), ("OTHER", "2024", "900")]]
    r = Retriever(chunk_blocks(blocks_from_documents(docs)))
    plan = {"operation": "sum", "operands": [{"doc_id": "ACME/2024/p", "row": 1, "column": 1}]}
    original = httpx.Client
    def respond(request):
        body = json.loads(request.content)
        prompt = json.loads(body["messages"][1]["content"])
        assert [t["doc_id"] for t in prompt["retrieved_tables"]] == ["ACME/2024/p"]
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(plan)}}]})
    monkeypatch.setattr(httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(respond)))
    planner = ModelPlanner("http://fixture/v1", "fixture")
    answer = planner.answer(r, docs, "ACME report year 2024 revenue")
    assert answer["value"] == "100"
    assert planner.answer(r, docs, "ACME revenue")["refused"]
    assert planner.answer(r, docs, "OTHER report year 2022 revenue")["refused"]
    assert planner.answer(r, docs, "revenue", scope_doc_id="ACME/2024/p")["value"] == "100"
