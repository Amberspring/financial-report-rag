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


def select_rows(source, blocked_ids, blocked_reports, count):
    eligible = []
    for row in source:
        doc_id = row.get("filename", row["id"].rsplit("-", 1)[0])
        if row["id"] not in blocked_ids and report_key(doc_id) not in blocked_reports:
            eligible.append((row["id"], row["qa"]["question"], doc_id))
    chosen = sorted(eligible, key=lambda row: (hashlib.sha256(row[0].encode()).hexdigest(), row[0]))[:count]
    if len(chosen) != count:
        raise ValueError(f"Fewer than {count} report-disjoint examples")
    return eligible, chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, help="Exclude all IDs and company/year reports from an earlier frozen holdout")
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("Count must be positive")
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve frozen holdouts")
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Official FinQA test source SHA-256 mismatch")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    previous_docs = json.loads((ROOT / "data/finqa-documents.json").read_text(encoding="utf-8"))
    previous_ids = {row["id"] for name in ("dev", "test") for row in json.loads(
        (ROOT / f"data/finqa-{name}-queries.json").read_text(encoding="utf-8"))}
    blocked_reports = {report_key(row["doc_id"]) for row in previous_docs}
    excluded_sha = None
    if args.exclude_manifest:
        excluded = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
        if excluded["source_sha256"] != SOURCE_SHA:
            raise ValueError("Excluded holdout uses a different source")
        previous_ids.update(excluded["ids"])
        blocked_reports.update(report_key(key) for key in excluded["report_counts"])
        excluded_sha = hashlib.sha256(args.exclude_manifest.read_bytes()).hexdigest()
    eligible, chosen = select_rows(source, previous_ids, blocked_reports, args.count)
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
        "selection": "Exclude prior dev/test IDs and every company/year in the 131 historical document pages" +
                     (" plus every ID and company/year in excluded manifest" if excluded_sha else "") +
                     "; take lowest SHA256(example_id), tie-break by ID; no answer/program inspected",
        "count": len(chosen), "eligible_before_selection": len(eligible),
        "distinct_company_year_reports": len(report_counts),
        "distinct_companies": len({key.split("/")[0] for key in report_counts}),
        "distinct_years": len({key.split("/")[1] for key in report_counts}),
        "report_counts": dict(sorted(report_counts.items())), "ids": ids,
        "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "answer_handling": "Gold remains in separately downloaded official source; never feed qa.program, qa.exe_ans, qa.gold_inds, qa.answer or correct program to prediction",
        "caveat": "Public test data may occur in pretraining; unseen only to this project's debugging selection.",
    }
    if excluded_sha:
        manifest["excluded_manifest_sha256"] = excluded_sha
    (args.output_dir / "holdout-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: manifest[k] for k in ("count", "eligible_before_selection", "distinct_company_year_reports", "distinct_companies", "distinct_years", "queries_sha256")}))


if __name__ == "__main__":
    main()
