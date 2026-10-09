"""Freeze FinQA report-disjoint IDs without inspecting answers or programs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc"


def report_key(doc_id):
    return tuple(doc_id.split("/")[:2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Official FinQA test source SHA-256 mismatch")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    previous_docs = json.loads((ROOT / "data/finqa-documents.json").read_text(encoding="utf-8"))
    previous_ids = {row["id"] for name in ("dev", "test") for row in json.loads(
        (ROOT / f"data/finqa-{name}-queries.json").read_text(encoding="utf-8"))}
    blocked_reports = {report_key(row["doc_id"]) for row in previous_docs}
    eligible = []
    for row in source:
        doc_id = row.get("filename", row["id"].rsplit("-", 1)[0])
        if row["id"] not in previous_ids and report_key(doc_id) not in blocked_reports:
            eligible.append((row["id"], row["qa"]["question"], doc_id))
    chosen = sorted(eligible, key=lambda row: (hashlib.sha256(row[0].encode()).hexdigest(), row[0]))[:200]
    if len(chosen) != 200:
        raise ValueError("Fewer than 200 report-disjoint examples")
    ids = [row[0] for row in chosen]
    report_counts = Counter("/".join(report_key(row[2])) for row in chosen)
    if len(report_counts) < 10 or len({key.split("/")[0] for key in report_counts}) < 5 or len({key.split("/")[1] for key in report_counts}) < 2:
        raise ValueError("Holdout diversity requirements not met")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    queries = [{"id": row_id, "given_report_question": question,
                "global_question": f"In the {doc_id.split('/')[0]} report year {doc_id.split('/')[1]}, {question}",
                "report_context_for_protocol_A": doc_id}
               for row_id, question, doc_id in chosen]
    queries_path = args.output_dir / "holdout-queries.json"
    queries_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "status": "frozen_ids_pending_independent_gold_review_and_model_run",
        "source": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
        "source_sha256": SOURCE_SHA, "upstream_split": "public_test",
        "license": "FinQA repository MIT; original annual-report rights need separate review",
        "selection": "Exclude prior dev/test IDs and every company/year in the 131 historical document pages; take lowest SHA256(example_id), tie-break by ID; no answer/program inspected",
        "count": len(chosen), "eligible_before_selection": len(eligible),
        "distinct_company_year_reports": len(report_counts),
        "distinct_companies": len({key.split("/")[0] for key in report_counts}),
        "distinct_years": len({key.split("/")[1] for key in report_counts}),
        "report_counts": dict(sorted(report_counts.items())), "ids": ids,
        "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "answer_handling": "Gold remains in separately downloaded official source; never feed qa.program, qa.exe_ans, qa.gold_inds, qa.answer or correct program to prediction",
        "caveat": "Public test data may occur in pretraining; unseen only to this project's debugging selection.",
    }
    (args.output_dir / "holdout-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: manifest[k] for k in ("count", "eligible_before_selection", "distinct_company_year_reports", "distinct_companies", "distinct_years", "queries_sha256")}))


if __name__ == "__main__":
    main()
