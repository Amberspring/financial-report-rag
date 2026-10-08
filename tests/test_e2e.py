from finrag.core import Retriever, chunk_blocks
from finrag.finqa import blocks_from_documents
from finrag.qa import numerical_answer


def test_global_calculation_requires_retrieved_unique_company_evidence():
    docs = [{"doc_id": "a", "company": "ACME", "report_year": "2024", "page": 1,
             "pre_text": [], "post_text": [], "table": [["metric", "2024", "2023"], ["Revenue", "120", "100"]]}]
    r = Retriever(chunk_blocks(blocks_from_documents(docs)))
    result = numerical_answer(r, docs, "What is ACME revenue growth from 2023 to 2024?")
    assert not result["refused"] and result["value"] == "20.0" and result["citations"]
    assert numerical_answer(r, docs, "What is OTHER revenue growth from 2023 to 2024?")["refused"]
    assert numerical_answer(r, docs, "What is revenue growth from 2023 to 2024?")["refused"]
    more = [{**docs[0], "doc_id": "b"}]
    r = Retriever(chunk_blocks(blocks_from_documents(docs + more)))
    assert numerical_answer(r, docs + more, "ACME revenue growth from 2023 to 2024")["refused"]
