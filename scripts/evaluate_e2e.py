"""Public FinQA end-to-end audit; labels used only AFTER prediction."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from finrag.core import Retriever, chunk_blocks
from finrag.finqa import documents, blocks_from_documents
from finrag.qa import numerical_answer
from finrag.planner import ModelPlanner


def run(output, embedding="tfidf", reranker=None, url=None, model=None):
    if bool(url) != bool(model):
        raise ValueError("Model and URL must be supplied together")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_path = target.with_suffix(".predictions.jsonl")
    if target.exists() or raw_path.exists():
        raise ValueError("Use a fresh output path to preserve experiment evidence")
    root = Path(__file__).resolve().parents[1]
    docs = documents(root / "data/finqa-documents.json")
    queries = json.loads((root / "data/finqa-test-queries.json").read_text(encoding="utf-8"))
    labels = {r["id"]: r for r in json.loads((root / "data/finqa-test-labels.json").read_text(encoding="utf-8"))}
    r = Retriever(chunk_blocks(blocks_from_documents(docs), 512, 50), embedding=embedding, reranker=reranker)
    experiments = []
    planner = ModelPlanner(url, model) if url and model else None
    modes = [("bm25", False)] if planner else [("dense", False), ("bm25", False), ("hybrid", False)] + ([("hybrid", True)] if reranker else [])
    for mode, rerank in modes:
        cases = []
        for q in queries:
            start = time.perf_counter()
            prediction = planner.answer(r, docs, q["query"], mode, rerank) if planner else numerical_answer(r, docs, q["query"], mode, rerank)
            elapsed = (time.perf_counter() - start) * 1000
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": q["id"], "mode": mode, "rerank": rerank, "latency_ms": elapsed, "prediction": prediction}, ensure_ascii=False) + "\n")
            label = labels[q["id"]]
            try:
                gold = float(label["exe_ans"])
            except (ValueError, TypeError):
                gold = None
            correct = gold is not None and not prediction["refused"] and abs(float(prediction["value"]) - gold) <= max(0.01, abs(gold) * 0.001)
            covered = set()
            for hit in prediction["hits"][:5]:
                if hit["chunk"]["doc_id"] == q["doc_id"]:
                    covered.update(hit["chunk"]["evidence_keys"])
            recall = len(covered & set(label["gold_inds"])) / max(1, len(label["gold_inds"]))
            cases.append({"id": q["id"], "correct_raw_exe": correct, "recall_at_5": recall,
                          "latency_ms": elapsed, "prediction": prediction, "gold_exe_ans": gold})
        n = len(cases)
        experiments.append({"mode": mode, "rerank": rerank, "n": n,
                            "coverage": sum(not c["prediction"]["refused"] for c in cases) / n,
                            "raw_execution_accuracy": sum(c["correct_raw_exe"] for c in cases) / n,
                            "recall_at_5": sum(c["recall_at_5"] for c in cases) / n,
                            "mean_e2e_ms": sum(c["latency_ms"] for c in cases) / n, "cases": cases})
    report = {"status": "measured", "scope": "global retrieval and narrow calculation, no gold doc_id passed to prediction",
              "embedding": embedding, "corpus_sha256": hashlib.sha256((root / "data/finqa-documents.json").read_bytes()).hexdigest(),
              "model": model, "planner": "retrieved-table model plan" if planner else "symbol-gated rule plan",
              "queries_sha256": hashlib.sha256((root / "data/finqa-test-queries.json").read_bytes()).hexdigest(),
              "labels_sha256": hashlib.sha256((root / "data/finqa-test-labels.json").read_bytes()).hexdigest(),
              "predictions_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
              "limitations": ["Public previously examined test subset; not a fresh unseen test", "Unsupported questions refused; single-operation table calculations only", "Raw exe_ans units retained; no implicit rescaling", "No GraphRAG claim"],
              "experiments": experiments}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in e.items() if k != "cases"} for e in experiments]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--embedding", default="tfidf")
    p.add_argument("--reranker")
    p.add_argument("--url")
    p.add_argument("--model")
    a = p.parse_args()
    run(a.output, a.embedding, a.reranker, a.url, a.model)
