"""Post-prediction diagnosis of the immutable 2026-10-08 FinQA run."""

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "results/finqa-model-20261008.predictions.jsonl"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def raw_match(prediction, label):
    if prediction.get("refused"):
        return False, False
    try:
        value = Decimal(str(prediction["value"]))
        gold = Decimal(str(label["exe_ans"]))
    except (KeyError, InvalidOperation, TypeError):
        return False, False
    tolerance = max(Decimal("0.01"), abs(gold) * Decimal("0.001"))
    historical = abs(value - gold) <= tolerance
    normalized = value / 100 if prediction.get("unit") == "percent" else value
    return historical, abs(normalized - gold) <= tolerance


def diagnose(query, label, prediction):
    target = query["doc_id"]
    company, report_year = target.split("/")[:2]
    hits = prediction.get("hits", [])
    ranks = [i + 1 for i, hit in enumerate(hits) if hit["chunk"]["doc_id"] == target]
    gold_keys = set(label["gold_inds"])
    evidence_5 = {key for hit in hits[:5] if hit["chunk"]["doc_id"] == target
                  for key in hit["chunk"]["evidence_keys"]}
    evidence_10 = {key for hit in hits if hit["chunk"]["doc_id"] == target
                   for key in hit["chunk"]["evidence_keys"]}
    trace = prediction.get("model_trace", {})
    prompt_docs = [table["doc_id"] for table in trace.get("request", {}).get("retrieved_tables", [])]
    operands = prediction.get("operands", [])
    selected_docs = sorted({operand["doc_id"] for operand in operands})
    gold_ops = re.findall(r"\b([a-z_]+)\(", label["program"])
    predicted_op = prediction.get("operation")
    old_match, normalized_match = raw_match(prediction, label)
    explicit_company = bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(company) + r"(?![A-Za-z0-9])", query["query"], re.I))
    explicit_report_year = bool(re.search(r"(?<!\d)" + re.escape(report_year) + r"(?!\d)", query["query"]))
    refused = bool(prediction.get("refused"))
    if not refused and old_match:
        category = "correct_historical"
    elif not refused and selected_docs != [target]:
        category = "wrong_report"
    elif not refused and normalized_match:
        category = "unit_or_scoring_mismatch"
    elif not refused:
        category = "operation_mismatch" if predicted_op not in gold_ops else "operand_or_multistep_mismatch"
    elif not ranks:
        category = "target_report_not_retrieved"
    elif not gold_keys & evidence_10:
        category = "gold_evidence_not_retrieved"
    elif target not in prompt_docs:
        category = "target_report_not_in_table_prompt"
    elif prediction.get("reason") == "Mixed cell units":
        category = "calculator_unit_conflict"
    elif prediction.get("reason"):
        category = "invalid_or_unsupported_model_plan"
    elif all(key.startswith("text_") for key in gold_keys):
        category = "prose_only_evidence_unsupported"
    elif len(gold_ops) > 1:
        category = "multi_step_or_planner_refusal"
    else:
        category = "planner_refusal_or_validation"
    return {
        "id": query["id"], "question": query["query"], "target_report_after_scoring_only": target,
        "query_names_target_company": explicit_company, "query_names_report_year": explicit_report_year,
        "historical_outcome": "refused" if refused else "correct" if old_match else "wrong_answer",
        "primary_observed_category": category, "diagnosis_status": "automated_evidence_triage_not_manual_pdf_review",
        "target_report_ranks_in_top10": ranks, "gold_evidence_keys": sorted(gold_keys),
        "gold_evidence_keys_top5": sorted(gold_keys & evidence_5),
        "gold_evidence_keys_top10": sorted(gold_keys & evidence_10),
        "gold_evidence_in_prose": any(key.startswith("text_") for key in gold_keys),
        "gold_evidence_in_table": any(key.startswith("table_") for key in gold_keys),
        "table_prompt_doc_ids": prompt_docs, "selected_operand_doc_ids": selected_docs,
        "selected_operands": operands, "predicted_operation": predicted_op, "gold_program_operations_diagnosis_only": gold_ops,
        "model_plan_text": (trace.get("response", {}).get("choices") or [{}])[0].get("message", {}).get("content"),
        "refusal_reason": prediction.get("reason"), "predicted_value": prediction.get("value"),
        "predicted_unit": prediction.get("unit"), "gold_exe_ans": label["exe_ans"],
        "gold_display_answer": label["answer"], "gold_program_diagnosis_only": label["program"],
        "historical_raw_match": old_match, "percent_normalized_diagnostic_match": normalized_match,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    queries_path = ROOT / "data/finqa-test-queries.json"
    labels_path = ROOT / "data/finqa-test-labels.json"
    queries = {row["id"]: row for row in read_json(queries_path)}
    labels = {row["id"]: row for row in read_json(labels_path)}
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    ids = [row["id"] for row in predictions]
    if len(ids) != 100 or len(set(ids)) != 100 or set(ids) != set(queries) or set(ids) != set(labels):
        raise ValueError("Historical 100 IDs must match queries and labels exactly")
    rows = [diagnose(queries[row["id"]], labels[row["id"]], row["prediction"]) for row in predictions]
    counts = Counter(row["historical_outcome"] for row in rows)
    if counts != Counter({"refused": 71, "wrong_answer": 27, "correct": 2}):
        raise ValueError(f"Historical outcome changed: {counts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = args.output_dir / "historical-100-ledger.jsonl"
    ledger.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
    report = {
        "scope": "Post-hoc historical global retrieval diagnosis; never a new model accuracy result",
        "historical_outcomes": dict(counts),
        "primary_observed_categories": dict(Counter(row["primary_observed_category"] for row in rows)),
        "query_names_target_company": sum(row["query_names_target_company"] for row in rows),
        "query_names_report_year": sum(row["query_names_report_year"] for row in rows),
        "target_report_in_top10": sum(bool(row["target_report_ranks_in_top10"]) for row in rows),
        "gold_evidence_in_top5": sum(bool(row["gold_evidence_keys_top5"]) for row in rows),
        "percent_normalized_diagnostic_matches": sum(row["percent_normalized_diagnostic_match"] for row in rows),
        "limitations": ["Gold labels and target document were used only after reading saved predictions.",
                        "Categories are observable triage, not a manual PDF or causal oracle verdict.",
                        "Percent normalization is a scoring diagnostic and does not alter the historical 2/100 result."],
        "inputs_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (queries_path, labels_path, args.predictions)},
        "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
