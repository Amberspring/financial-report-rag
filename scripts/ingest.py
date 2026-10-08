import argparse, json
from pathlib import Path
from dataclasses import asdict
from finrag.core import parse_pdf, chunk_blocks

p = argparse.ArgumentParser()
p.add_argument("pdf")
p.add_argument("--output", default="artifacts/parsed.json")
a = p.parse_args()
blocks = parse_pdf(a.pdf)
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps([asdict(b) for b in blocks], ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "blocks": len(blocks),
            "tables": sum(b.kind == "table" for b in blocks),
            "chunks": len(chunk_blocks(blocks)),
        }
    )
)
