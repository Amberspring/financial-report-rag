"""Re-evaluate narrow planner; preserve raw exe_ans and human-readable units separately."""

import json, re
from pathlib import Path
from finrag.finqa import documents
from finrag.calculator import propose, execute, number, CalculationError

root = Path(__file__).resolve().parents[1]
docs = documents(root / "data/finqa-documents.json")
lookup = {d["doc_id"]: d for d in docs}
queries = json.loads(
    (root / "data/finqa-test-queries.json").read_text(encoding="utf-8")
)
labels = {
    r["id"]: r
    for r in json.loads(
        (root / "data/finqa-test-labels.json").read_text(encoding="utf-8")
    )
}
rows = []
for q in queries:
    row = {"id": q["id"], "query": q["query"]}
    try:
        pred = execute(docs, propose(q["query"], lookup[q["doc_id"]]))
        value = float(pred["value"])
        label = labels[q["id"]]
        gold = float(label["exe_ans"])
        strict = abs(value - gold) <= max(0.01, abs(gold) * 0.001)
        presented = None
        presented_gold = None
        try:
            human, unit = number(label["answer"])
            if (
                unit == pred["unit"]
                or unit == "reported_units"
                and pred["unit"] == "reported_units"
            ):
                presented_gold = float(human)
                fraction = str(label["answer"]).split(".", 1)
                decimals = (
                    len(re.sub(r"\D", "", fraction[1])) if len(fraction) == 2 else 0
                )
                tolerance = (
                    10 ** (-decimals)
                    if unit == "percent"
                    else max(0.01, abs(presented_gold) * 0.001)
                )
                presented = abs(value - presented_gold) <= tolerance
        except CalculationError:
            pass
        row.update(
            answered=True,
            prediction=pred,
            strict_raw_exe_match=strict,
            human_answer_match=presented,
            gold_exe_ans=gold,
            gold_answer=label["answer"],
            gold_program=label["program"],
            human_numeric=presented_gold,
        )
    except CalculationError as e:
        row.update(answered=False, reason=str(e))
    rows.append(row)
answered = [r for r in rows if r["answered"]]
comparable = [r for r in answered if r["human_answer_match"] is not None]
report = {
    "status": "measured",
    "version": "post_bugfix_basis_points_rejected",
    "n": len(rows),
    "answered": len(answered),
    "coverage": len(answered) / len(rows),
    "raw_exe_match_count": sum(r["strict_raw_exe_match"] for r in answered),
    "human_answer_match_count": sum(bool(r["human_answer_match"]) for r in comparable),
    "human_answer_comparable_n": len(comparable),
    "scope": "document supplied; not global end-to-end QA",
    "metric_notes": [
        "Raw exe_ans can express ratios while printed answers express percentages. Both comparisons retained; no silent rescaling.",
        "Printed percentage tolerance is one unit in its last shown decimal, accommodating display truncation.",
        "Human match is only a label/units audit, not standard FinQA execution accuracy.",
        "Basis-point convention bug was found after initial test evaluation and then fixed; this is re-evaluation on the same public test subset, not a new unseen test.",
    ],
    "cases": rows,
}
(root / "results/numerical-audit.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps({k: v for k, v in report.items() if k != "cases"}))
