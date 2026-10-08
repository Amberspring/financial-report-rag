import pytest
from finrag.calculator import execute, propose, CalculationError
from finrag.finqa import blocks_from_documents
from finrag.core import chunk_blocks

DOC = {
    "doc_id": "ACME/2024/page_5.pdf",
    "company": "ACME",
    "report_year": "2024",
    "page": 5,
    "pre_text": ["Revenue was audited."],
    "post_text": ["No guarantee."],
    "table": [
        ["USD million", "2024", "2023"],
        ["Revenue", "$120.5", "$100.0"],
        ["Net income", "20", "10"],
        ["Margin", "20%", "10%"],
    ],
}


def refs(row=1):
    return [
        {"doc_id": DOC["doc_id"], "row": row, "column": 2, "expected_year": "2023"},
        {"doc_id": DOC["doc_id"], "row": row, "column": 1, "expected_year": "2024"},
    ]


def test_evidence_decimal_and_period_trace():
    result = execute(
        [DOC], {"operation": "subtract", "operands": list(reversed(refs()))}
    )
    assert (
        result["value"] == "20.5"
        and result["operands"][0]["page"] == 5
        and result["citations"][0]["quote"] == "$120.5"
    )
    plan = propose("What is the percentage change in revenue from 2023 to 2024?", DOC)
    assert execute([DOC], plan)["value"] == "20.500"


@pytest.mark.parametrize("mutation", ["year", "row", "label", "code", "unit", "zero"])
def test_bad_operands_fail_closed(mutation):
    import copy

    doc = copy.deepcopy(DOC)
    plan = {"operation": "subtract", "operands": refs()}
    if mutation == "year":
        plan["operands"][0]["expected_year"] = "2022"
    if mutation == "row":
        plan["operands"][0]["row"] = -1
    if mutation == "label":
        plan["operands"][0]["expected_label"] = "cash flow"
    if mutation == "code":
        plan["operation"] = '__import__("os")'
    if mutation == "unit":
        plan["operands"][0]["row"] = 3
    if mutation == "zero":
        doc["table"][1][1] = "0"
        plan["operation"] = "divide"
    with pytest.raises(CalculationError):
        execute([doc], plan)


def test_inference_blocks_never_contain_gold_annotations():
    blocks = blocks_from_documents([DOC])
    chunks = chunk_blocks(blocks, 20, 5)
    assert all("program" not in c.text for c in chunks)
    assert next(c for c in chunks if c.kind == "table").evidence_keys == (
        "table_1",
        "table_2",
        "table_3",
    )
    assert len(next(c for c in chunks if c.kind == "table").text) > 20


def test_basis_points_are_not_mistaken_for_growth():
    with pytest.raises(CalculationError):
        propose("What is the change in basis points of margin from 2023 to 2024?", DOC)


def test_mixed_currencies_are_rejected():
    import copy

    doc = copy.deepcopy(DOC)
    doc["table"][1][2] = "€100.0"
    with pytest.raises(CalculationError):
        execute([doc], {"operation": "subtract", "operands": refs()})


def test_invalid_metadata_types_raise_calculation_errors():
    for ref in [
        {"doc_id": [], "row": 1, "column": 1},
        {"doc_id": DOC["doc_id"], "row": 1, "column": 1, "expected_label": 42},
    ]:
        with pytest.raises(CalculationError):
            execute([DOC], {"operation": "sum", "operands": [ref]})
