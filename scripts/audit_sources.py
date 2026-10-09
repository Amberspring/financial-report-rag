"""Verify every included FinQA record against official source bytes and manifest hashes."""
import hashlib
import json
from pathlib import Path
import httpx


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/finqa-manifest.json").read_text(encoding="utf-8"))
    reports, official = [], {}
    for split in ("dev", "test"):
        src = next(r for r in manifest["source_files"] if r.get("split") == split + ".json")
        url = src.get("url", src.get("source"))
        response = httpx.get(url, follow_redirects=True, timeout=120)
        response.raise_for_status()
        digest = hashlib.sha256(response.content).hexdigest()
        if digest != src["sha256"]:
            raise ValueError("Official data fingerprint changed")
        records = {r["id"]: r for r in response.json()}
        official.update(records)
        queries = json.loads((root / f"data/finqa-{split}-queries.json").read_text(encoding="utf-8"))
        labels = {r["id"]: r for r in json.loads((root / f"data/finqa-{split}-labels.json").read_text(encoding="utf-8"))}
        for query in queries:
            original, label = records[query["id"]], labels[query["id"]]
            if query["query"] != original["qa"]["question"] or not all(label[k] == original["qa"][k] for k in ("gold_inds", "exe_ans", "program", "answer")):
                raise ValueError("Question or label mismatch: " + query["id"])
        reports.append({"split": split, "url": url, "sha256": digest, "records_verified": len(queries)})
    docs = json.loads((root / "data/finqa-documents.json").read_text(encoding="utf-8"))
    by_doc = {}
    for record in official.values():
        by_doc.setdefault(record.get("filename", record["id"].rsplit("-", 1)[0]), []).append(record)
    for doc in docs:
        if not any(doc["pre_text"] == r["pre_text"] and doc["post_text"] == r["post_text"] and doc["table"] == (r.get("table_ori") or r["table"]) for r in by_doc[doc["doc_id"]]):
            raise ValueError("Document mismatch: " + doc["doc_id"])
    output = {"status": "verified_against_official_source", "documents_verified": len(docs), "sources": reports,
              "limitation": "FinQA transcriptions verified; original PDF page numbering and correctness of authors' labels not independently certified"}
    (root / "results/source-audit-20261008.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
