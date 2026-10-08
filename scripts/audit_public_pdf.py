"""Download a real public PDF and audit extraction. Original PDF is excluded from Git."""

import argparse, hashlib, json, time
from pathlib import Path
import httpx, pdfplumber
from finrag.core import parse_pdf, chunk_blocks

p = argparse.ArgumentParser()
p.add_argument(
    "--url",
    default="https://microsoft.gcs-web.com/static-files/1c864583-06f7-40cc-a94d-d11400c83cc8",
)
a = p.parse_args()
out = Path("artifacts/public-pdf")
out.mkdir(parents=True, exist_ok=True)
pdf = out / "msft-2024-10k.pdf"
if not pdf.exists():
    response = httpx.get(a.url, follow_redirects=True, timeout=120)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("Not a PDF")
    pdf.write_bytes(response.content)
start = time.perf_counter()
blocks = parse_pdf(pdf, "MSFT-2024-10K")
chunks = chunk_blocks(blocks)
with pdfplumber.open(pdf) as source:
    pages = len(source.pages)
report = {
    "status": "measured",
    "source_url": a.url,
    "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    "pages": pages,
    "blocks": len(blocks),
    "table_blocks": sum(b.kind == "table" for b in blocks),
    "chunks": len(chunks),
    "seconds": time.perf_counter() - start,
    "warning": "Extraction counts do not establish OCR, reading-order, or table-cell accuracy; manual QA required. FinQA benchmark is a separate corpus.",
}
Path("results").mkdir(exist_ok=True)
Path("results/real-pdf-audit.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report))
