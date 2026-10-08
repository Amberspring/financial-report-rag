"""FinQA input loader. Never reads questions' labels or gold programs."""

import json
from pathlib import Path
from .core import Block


def documents(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def blocks_from_documents(rows):
    blocks = []
    for d in rows:
        prefix = f"Company: {d['company']}; report year: {d['report_year']}; page: {d['page']}. "
        for i, text in enumerate(d["pre_text"] + d["post_text"]):
            blocks.append(
                Block(
                    d["doc_id"], d["page"], prefix + text, evidence_keys=(f"text_{i}",)
                )
            )
        table = d["table"]
        text = "\n".join(" | ".join(map(str, row)) for row in table)
        blocks.append(
            Block(
                d["doc_id"],
                d["page"],
                prefix + text,
                "table",
                evidence_keys=tuple(f"table_{i}" for i in range(1, len(table))),
            )
        )
    return blocks
