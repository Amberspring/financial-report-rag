from pathlib import Path
import json, hashlib, re

research = Path(__file__).resolve().parents[1]
cache = research / "artifacts/finqa-source"
cache.mkdir(parents=True, exist_ok=True)
import httpx

manifest = []
expected = json.loads(
    (research / "data/finqa-manifest.json").read_text(encoding="utf-8")
)["source_files"]
for split in ("dev", "test"):
    url = (
        "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/"
        + split
        + ".json"
    )
    response = httpx.get(url, follow_redirects=True, timeout=120)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    old = next((r for r in expected if r.get("url", r.get("source")) == url), None)
    if old and old["sha256"] != digest:
        raise ValueError(
            "Upstream data changed; review before updating the pinned benchmark"
        )
    (cache / (split + ".json")).write_bytes(response.content)
    manifest.append({"url": url, "rows": len(response.json()), "sha256": digest})
(cache / "manifest.json").write_text(json.dumps(manifest, indent=2))
chosen = {}
for split, count in [("dev", 40), ("test", 100)]:
    values = json.loads((cache / (split + ".json")).read_text(encoding="utf-8"))
    values = sorted(values, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[
        :count
    ]
    chosen[split] = values
docs = {}
labels = {}
queries = {}
for split, values in chosen.items():
    labels[split] = []
    queries[split] = []
    for r in values:
        doc_id = r.get("filename", r["id"].rsplit("-", 1)[0])
        page = int(re.search(r"page_(\d+)", doc_id).group(1))
        docs[doc_id] = {
            "doc_id": doc_id,
            "company": doc_id.split("/")[0],
            "report_year": doc_id.split("/")[1],
            "page": page,
            "pre_text": r["pre_text"],
            "post_text": r["post_text"],
            "table": r.get("table_ori") or r["table"],
            "source": "https://github.com/czyssrs/FinQA",
            "source_type": "public_benchmark_transcription_not_downloaded_PDF",
        }
        queries[split].append(
            {"id": r["id"], "query": r["qa"]["question"], "doc_id": doc_id}
        )
        labels[split].append(
            {
                "id": r["id"],
                "gold_inds": r["qa"]["gold_inds"],
                "exe_ans": r["qa"]["exe_ans"],
                "program": r["qa"]["program"],
                "answer": r["qa"]["answer"],
            }
        )
(research / "data/finqa-documents.json").write_text(
    json.dumps(list(docs.values()), ensure_ascii=False, indent=2), encoding="utf-8"
)
for split in chosen:
    (research / "data" / f"finqa-{split}-queries.json").write_text(
        json.dumps(queries[split], indent=2), encoding="utf-8"
    )
    (research / "data" / f"finqa-{split}-labels.json").write_text(
        json.dumps(labels[split], indent=2), encoding="utf-8"
    )
(research / "data/finqa-manifest.json").write_text(
    json.dumps(
        {
            "selection": "lowest SHA256(example_id); 40 development +100 test; official split retained",
            "source_files": json.loads(
                (cache / "manifest.json").read_text(encoding="utf-8")
            ),
            "documents": len(docs),
            "queries": {s: len(v) for s, v in queries.items()},
            "no_label_inputs": [
                "documents",
                "questions",
                "tables",
                "pre_text",
                "post_text",
            ],
            "excluded_from_inference": [
                "program",
                "exe_ans",
                "answer",
                "gold_inds",
                "model_input",
                "table_retrieved",
                "text_retrieved",
            ],
        },
        indent=2,
    ),
    encoding="utf-8",
)
