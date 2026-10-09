import importlib.util
import json
from pathlib import Path
from finrag.core import Block


def test_pdf_audit_preserves_source_and_physical_page(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/audit_pdf.py"
    spec = importlib.util.spec_from_file_location("audit_pdf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "parse_pdf", lambda *a: [Block("fixture", 3, "contractual obligations total 123")])
    source = tmp_path / "fixture.pdf"
    source.write_bytes(b"not a real PDF; parser mocked")
    result = module.run(source, "https://fixture/report.pdf", "contractual obligations", tmp_path / "audit")
    assert result["source"] == "https://fixture/report.pdf" and result["pages_with_extracted_blocks"] == 1
    assert result["experiments"]["bm25"][0]["chunk"]["page"] == 3
    assert json.loads((tmp_path / "audit/audit.json").read_text())["status"] == "measured_pdf_ingestion_and_retrieval"
