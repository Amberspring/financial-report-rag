from finrag.core import Block, chunk_blocks, Retriever, answer, parse_pdf
import pytest


def test_table_atomic_and_pages():
    blocks = [Block("a", 3, "x" * 600, "table"), Block("a", 4, "y" * 600)]
    chunks = chunk_blocks(blocks, 256, 50)
    assert len([c for c in chunks if c.kind == "table"]) == 1
    assert len(chunks[0].text) == 600 and chunks[0].page == 3
    assert all(len(c.text) <= 256 for c in chunks if c.kind == "text")
    with pytest.raises(ValueError):
        chunk_blocks(blocks, 10, 10)


def test_retrieval_and_refusal():
    r = Retriever(
        chunk_blocks(
            [
                Block("sales", 1, "星河科技营业收入为100亿元。"),
                Block("risk", 2, "海风能源面临原材料风险。"),
            ]
        )
    )
    for mode in ("dense", "bm25", "hybrid"):
        assert r.search("星河科技营业收入", 1, mode)[0]["chunk"]["doc_id"] == "sales"
    result = answer(r, "星河科技营业收入")
    assert result["citations"][0]["page"] == 1
    assert answer(r, "火星旅行天气")["refused"]


def test_pdf_table_extraction():
    from pathlib import Path

    blocks = parse_pdf(Path(__file__).parent.parent / "data/synthetic-report.pdf")
    assert any(
        b.kind == "table" and "Revenue" in b.text and "100" in b.text for b in blocks
    )
    assert not any("Revenue" in b.text for b in blocks if b.kind == "text")


def test_index_variants():
    chunks = chunk_blocks(
        [Block("a", 1, "alpha profit"), Block("b", 1, "beta revenue")]
    )
    for kind in ("flat", "ivf", "hnsw"):
        assert (
            Retriever(chunks, index_type=kind).search("alpha profit", 1, "dense")[0][
                "chunk"
            ]["doc_id"]
            == "a"
        )


def test_trusted_index_roundtrip(tmp_path):
    r = Retriever(
        chunk_blocks(
            [Block("a", 1, "营业收入100亿元"), Block("b", 2, "原材料价格风险")]
        )
    )
    r.save(tmp_path)
    loaded = Retriever.load_trusted(tmp_path)
    assert loaded.search("营业收入", 1)[0]["chunk"]["doc_id"] == "a"


def test_irrelevant_prediction_and_missing_year_refused():
    r = Retriever(
        chunk_blocks(
            [Block("a", 1, "星河科技2025年收入100亿元。行业发展不构成业绩预测。")]
        )
    )
    assert answer(r, "火星旅游气温预测")["refused"]
    assert answer(r, "星河科技2029年收入")["refused"]
