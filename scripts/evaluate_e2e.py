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
from finrag.scoring import execution_match


def run(output, embedding="tfidf", reranker=None, url=None, model=None, protocol="global",
        documents_path=None, queries_path=None, labels_path=None):
    if bool(url) != bool(model):
        raise ValueError("Model and URL must be supplied together")
    if protocol not in ("global", "given_report") or protocol == "given_report" and not url:
        raise ValueError("Given-report protocol requires a model URL; choose an explicit protocol")
    if any(p is not None for p in (documents_path, queries_path, labels_path)) and not all(
        p is not None for p in (documents_path, queries_path, labels_path)):
        raise ValueError("Supply documents, queries and labels together")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_path = target.with_suffix(".predictions.jsonl")
    if target.exists() or raw_path.exists():
        raise ValueError("Use a fresh output path to preserve experiment evidence")
    root = Path(__file__).resolve().parents[1]
    documents_path = Path(documents_path) if documents_path else root / "data/finqa-documents.json"
    queries_path = Path(queries_path) if queries_path else root / "data/finqa-test-queries.json"
    labels_path = Path(labels_path) if labels_path else root / "data/finqa-test-labels.json"
    docs = documents(documents_path)
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    r = Retriever(chunk_blocks(blocks_from_documents(docs), 512, 50), embedding=embedding, reranker=reranker)
    predictions_by_mode = []
    planner = ModelPlanner(url, model) if url and model else None
    modes = [("bm25", False)] if planner else [("dense", False), ("bm25", False), ("hybrid", False)] + ([("hybrid", True)] if reranker else [])
    for mode, rerank in modes:
        cases = []
        for q in queries:
            start = time.perf_counter()
            prediction = planner.answer(r, docs, q["query"], mode, rerank,
                                        scope_doc_id=q["doc_id"] if protocol == "given_report" else None) if planner else numerical_answer(r, docs, q["query"], mode, rerank)
            elapsed = (time.perf_counter() - start) * 1000
            with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"id": q["id"], "mode": mode, "rerank": rerank, "latency_ms": elapsed, "prediction": prediction}, ensure_ascii=False) + "\n")
            cases.append({"id": q["id"], "doc_id": q["doc_id"], "latency_ms": elapsed, "prediction": prediction})
        predictions_by_mode.append((mode, rerank, cases))
    # Gold labels enter only after every model prediction has been written.
    labels = {r["id"]: r for r in json.loads(labels_path.read_text(encoding="utf-8"))}
    experiments = []
    for mode, rerank, raw_cases in predictions_by_mode:
        cases = []
        for raw in raw_cases:
            label = labels[raw["id"]]
            prediction = raw["prediction"]
            try:
                gold = float(label["exe_ans"])
            except (ValueError, TypeError):
                gold = None
            correct, score_status = execution_match(prediction, label)
            covered = set()
            for hit in prediction["hits"][:5]:
                if hit["chunk"]["doc_id"] == raw["doc_id"]:
                    covered.update(hit["chunk"]["evidence_keys"])
            recall = len(covered & set(label["gold_inds"])) / max(1, len(label["gold_inds"]))
            cases.append({"id": raw["id"], "correct_execution": correct, "score_status": score_status, "recall_at_5": recall,
                          "latency_ms": raw["latency_ms"], "answered": not prediction["refused"], "gold_exe_ans": gold})
        n = len(cases)
        experiments.append({"mode": mode, "rerank": rerank, "n": n,
                            "coverage": sum(c["answered"] for c in cases) / n,
                            "execution_accuracy": sum(c["correct_execution"] for c in cases) / n,
                            "recall_at_5": sum(c["recall_at_5"] for c in cases) / n,
                            "mean_e2e_ms": sum(c["latency_ms"] for c in cases) / n, "cases": cases})
    report = {"status": "measured", "protocol": protocol,
              "scope": "Question-only global retrieval" if protocol == "global" else "FinQA-provided report context; not a global retrieval result",
              "embedding": embedding, "corpus_sha256": hashlib.sha256(documents_path.read_bytes()).hexdigest(),
              "model": model, "planner": "retrieved-table model plan" if planner else "symbol-gated rule plan",
              "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
              "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
              "predictions_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
              "limitations": ["Public previously examined test subset; not a fresh unseen test", "Unsupported questions refused; single-operation table calculations only", "Percent outputs converted to FinQA raw fraction units for scoring", "No GraphRAG claim"],
              "experiments": experiments}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps([{k: v for k, v in e.items() if k != "cases"} for e in experiments]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--embedding", default="tfidf")
    p.add_argument("--reranker")
    p.add_argument("--url")
    p.add_argument("--model")
    p.add_argument("--protocol", choices=("global", "given_report"), default="global")
    p.add_argument("--documents")
    p.add_argument("--queries")
    p.add_argument("--labels")
    a = p.parse_args()
    run(a.output, a.embedding, a.reranker, a.url, a.model, a.protocol, a.documents, a.queries, a.labels)
