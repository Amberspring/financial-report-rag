import pytest

from scripts.diagnose_pipeline import diagnose


def test_posthoc_pipeline_stages_are_distinct():
    queries = [{"id": "q1", "doc_id": "right"}, {"id": "q2", "doc_id": "right"}]
    labels = [{"id": "q1", "gold_inds": {"table_1": ""}, "exe_ans": 20},
              {"id": "q2", "gold_inds": {"text_0": ""}, "exe_ans": 10}]
    rows = [
        {"id": "q1", "mode": "bm25", "prediction": {"refused": False, "value": "20", "unit": "reported_units",
            "operands": [{"doc_id": "right", "row": 1}],
            "hits": [{"chunk": {"doc_id": "right", "evidence_keys": ["table_1"]}}]}},
        {"id": "q2", "mode": "bm25", "prediction": {"refused": True,
            "hits": [{"chunk": {"doc_id": "wrong", "evidence_keys": ["text_0"]}}]}},
    ]
    result = diagnose(rows, queries, labels)[0]
    assert result["counts"]["target_report_top1"] == 1
    assert result["counts"]["gold_evidence_all_top5"] == 1
    assert result["counts"]["gold_evidence_selected_proxy"] == 1
    assert result["end_to_end_accuracy"] == .5
    assert result["accuracy_given_answer"] == 1
    with pytest.raises(ValueError, match="Incomplete"):
        diagnose(rows[:1], queries, labels)
