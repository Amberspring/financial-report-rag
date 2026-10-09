"""Real PDF ingestion and retrieval trace, without inventing answer labels."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from finrag.core import parse_pdf, chunk_blocks, Retriever


def run(pdf, source, query, output):
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    blocks = parse_pdf(pdf, Path(pdf).stem)
    chunks = chunk_blocks(blocks)
    retriever = Retriever(chunks)
    records = [asdict(b) for b in blocks]
    block_path = target / "blocks.json"
    block_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"status": "measured_pdf_ingestion_and_retrieval", "source": source,
              "pdf_sha256": hashlib.sha256(Path(pdf).read_bytes()).hexdigest(),
              "blocks_sha256": hashlib.sha256(block_path.read_bytes()).hexdigest(),
              "pages_with_extracted_blocks": len({b.page for b in blocks}), "blocks_by_kind": dict(Counter(b.kind for b in blocks)),
              "chunks": len(chunks), "query": query,
              "experiments": {mode: retriever.search(query, 5, mode) for mode in ("dense", "bm25", "hybrid")},
              "limitations": "Physical PDF page numbers; printed page labels may differ. TF-IDF dense baseline. Extracted evidence is not an answer accuracy score. No claim that automatic PDF tables are fully correct; visually verify quoted pages."}
    (target / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "experiments"}))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.pdf, a.source, a.query, a.output)
