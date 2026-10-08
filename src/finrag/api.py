import os, json
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .core import Block, chunk_blocks, Retriever, answer
from .finqa import documents, blocks_from_documents
from .calculator import execute, propose, CalculationError
from .qa import numerical_answer
from .planner import ModelPlanner

root = Path(__file__).resolve().parents[2]
retriever = None
docs = []
config = {}


def get_retriever():
    global retriever, docs, config
    if retriever is None:
        rows = documents(os.getenv("RAG_DATA", str(root / "data/finqa-documents.json")))
        docs = rows if rows and "table" in rows[0] else []
        blocks = blocks_from_documents(docs) if docs else [Block(**row) for row in rows]
        config = json.loads(
            Path(os.getenv("RAG_CONFIG", str(root / "configs/default.json"))).read_text(
                encoding="utf-8"
            )
        )
        config["embedding"] = os.getenv("EMBEDDING", config["embedding"])
        config["reranker"] = os.getenv("RERANKER", config.get("reranker") or "") or None
        retriever = Retriever(
            chunk_blocks(blocks, config["chunk_size"], config["overlap"]),
            embedding=config["embedding"],
            index_type=config["index"],
            reranker=config["reranker"],
        )
    return retriever


@asynccontextmanager
async def lifespan(app):
    get_retriever()
    yield


app = FastAPI(title="Financial Research Copilot", version="0.2.0", lifespan=lifespan)


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: str = Field(default="hybrid", pattern="^(dense|bm25|hybrid)$")
    doc_id: str | None = None
    calculate: bool = False


class Calculation(BaseModel):
    operation: str
    operands: list[dict] = Field(min_length=1, max_length=12)
    allow_cross_company: bool = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": retriever is not None,
        "backend": config.get("embedding"),
        "corpus": "FinQA public transcriptions" if docs else "user-supplied PDF blocks",
        "documents": len({c.doc_id for c in retriever.chunks}) if retriever else 0,
    }


@app.get("/documents")
def list_documents():
    get_retriever()
    return [
        {k: d[k] for k in ("doc_id", "company", "report_year", "page", "source")}
        for d in docs
    ]


@app.get("/documents/{doc_id:path}")
def document(doc_id: str):
    get_retriever()
    selected = [d for d in docs if d["doc_id"] == doc_id]
    if not selected:
        raise HTTPException(404, "Document not found")
    return selected[0]


@app.get("/example")
def example():
    return json.loads(
        (root / "data/calculation-example.json").read_text(encoding="utf-8")
    )


@app.post("/calculate")
def calculate(body: Calculation):
    get_retriever()
    try:
        return execute(docs, body.model_dump())
    except CalculationError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/query")
def query(body: Query):
    r = get_retriever()
    if body.calculate:
        if body.doc_id is None:
            if os.getenv("RAG_MODEL_URL") and os.getenv("RAG_MODEL"):
                return ModelPlanner(os.environ["RAG_MODEL_URL"], os.environ["RAG_MODEL"]).answer(r, docs, body.query, body.mode, bool(config.get("reranker")))
            return numerical_answer(r, docs, body.query, body.mode, bool(config.get("reranker")))
        selected = [d for d in docs if d["doc_id"] == body.doc_id]
        if len(selected) != 1:
            return {
                "answer": "A unique document is required for numerical planning.",
                "refused": True,
                "citations": [],
            }
        try:
            value = execute(docs, propose(body.query, selected[0]))
            return {
                **value,
                "answer": value["value"] + " " + value["unit"],
                "degraded": False,
            }
        except CalculationError as e:
            return {
                "answer": "Unsupported or ambiguous calculation: " + str(e),
                "refused": True,
                "citations": [],
            }
    if body.doc_id:
        chunks = [c for c in r.chunks if c.doc_id == body.doc_id]
        if not chunks:
            return {"answer": "Document not found.", "refused": True, "citations": []}
        # Explicit scope is user-provided; never inferred from a gold label in this API.
        scoped = Retriever(chunks, embedding="tfidf")
        return answer(scoped, body.query, body.mode, config["refusal_threshold"])
    return answer(
        r,
        body.query,
        body.mode,
        float(os.getenv("REFUSAL_THRESHOLD", str(config["refusal_threshold"]))),
        bool(config.get("reranker")),
    )


@app.get("/")
def demo():
    from fastapi.responses import HTMLResponse

    return HTMLResponse(
        """<!doctype html><meta charset="utf-8"><title>Financial Research Copilot</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto}input{padding:10px;width:75%}button{padding:10px}pre{white-space:pre-wrap;background:#f3f5f7;padding:20px}</style><h1>Financial Research Copilot</h1><p>公开 FinQA 财报转录数据。答案为证据摘录，支持页码引用和单独的受控计算接口。资料不足会拒答。</p><input id="q" value="What are the contractual obligations for HOLX in 2008?"><button onclick="run()">检索</button><pre id="out"></pre><script>async function run(){let r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:document.getElementById('q').value})});out.textContent=JSON.stringify(await r.json(),null,2)}</script>"""
    )
