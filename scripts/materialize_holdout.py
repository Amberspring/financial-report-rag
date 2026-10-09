"""Rebuild frozen report-page corpus, protocol queries and separate scoring labels."""

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def write(path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Source SHA-256 mismatch")
    source = {row["id"]: row for row in json.loads(args.source.read_text(encoding="utf-8"))}
    frozen = json.loads((args.manifest.parent / "holdout-queries.json").read_text(encoding="utf-8"))
    if [row["id"] for row in frozen] != manifest["ids"]:
        raise ValueError("Frozen query IDs differ from manifest")
    if hashlib.sha256((args.manifest.parent / "holdout-queries.json").read_bytes()).hexdigest() != manifest["queries_sha256"]:
        raise ValueError("Frozen query file changed")
    docs, a_queries, b_queries, labels = {}, [], [], []
    for item in frozen:
        row = source[item["id"]]
        doc_id = row.get("filename", row["id"].rsplit("-", 1)[0])
        if doc_id != item["report_context_for_protocol_A"]:
            raise ValueError("Report context changed")
        docs[doc_id] = {"doc_id": doc_id, "company": doc_id.split("/")[0],
                        "report_year": doc_id.split("/")[1],
                        "page": int(re.search(r"page_(\d+)", doc_id).group(1)),
                        "pre_text": row["pre_text"], "post_text": row["post_text"],
                        "table": row.get("table_ori") or row["table"],
                        "source": manifest["source"],
                        "source_type": "public_benchmark_page_transcription_not_full_report_pdf"}
        a_queries.append({"id": item["id"], "query": item["given_report_question"], "doc_id": doc_id})
        b_queries.append({"id": item["id"], "query": item["global_question"], "doc_id": doc_id})
        labels.append({"id": item["id"], **{key: row["qa"].get(key) for key in ("gold_inds", "exe_ans", "program", "answer")}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {name: write(args.output_dir / name, rows) for name, rows in (
        ("documents.json", list(docs.values())), ("queries-A.json", a_queries),
        ("queries-B-proxy.json", b_queries), ("labels-scoring-only.json", labels))}
    write(args.output_dir / "materialized-manifest.json", {
        "source_sha256": manifest["source_sha256"], "frozen_queries_sha256": manifest["queries_sha256"],
        "files_sha256": hashes, "documents": len(docs), "questions": len(a_queries),
        "warning": "Protocol B proxy uses selected FinQA page transcriptions, not complete annual-report PDFs or independently reviewed business gold."})
    print(json.dumps({"documents": len(docs), "questions": len(a_queries), "files_sha256": hashes}))


if __name__ == "__main__":
    main()
