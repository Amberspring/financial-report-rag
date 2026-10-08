"""Label-isolated public benchmark. Corpus/query inputs contain no gold answer annotations."""

import argparse, json, time, platform
from pathlib import Path
import numpy as np
from finrag.core import Retriever, chunk_blocks
from finrag.finqa import documents, blocks_from_documents
from finrag.calculator import propose, execute, CalculationError


def run():
    p = argparse.ArgumentParser()
    p.add_argument("--bge", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--neural", action="store_true")
    p.add_argument(
        "--flag-reranker",
        action="store_true",
        help="Use FlagEmbedding BGE reranker rather than the CPU CrossEncoder",
    )
    a = p.parse_args()
    import torch

    torch.set_num_threads(2)
    docs = documents("data/finqa-documents.json")
    blocks = blocks_from_documents(docs)
    queries = json.loads(
        Path("data/finqa-test-queries.json").read_text(encoding="utf-8")
    )
    labels = {
        x["id"]: x
        for x in json.loads(
            Path("data/finqa-test-labels.json").read_text(encoding="utf-8")
        )
    }
    Path("results").mkdir(exist_ok=True)
    report = {
        "status": "measured",
        "dataset": "FinQA official public test subset",
        "query_n": len(queries),
        "documents": len(docs),
        "platform": platform.platform(),
        "label_inputs": False,
        "experiments": [],
    }

    def evaluate(r, mode, rerank=False):
        traces = []
        times = []
        for q in queries:
            start = time.perf_counter()
            hits = r.search(q["query"], 10, mode, rerank)
            times.append((time.perf_counter() - start) * 1000)
            gold = set(labels[q["id"]]["gold_inds"])
            recalls = {}
            for k in (3, 5, 10):
                covered = set()
                for h in hits[:k]:
                    if h["chunk"]["doc_id"] == q["doc_id"]:
                        covered.update(h["chunk"]["evidence_keys"])
                recalls[k] = len(covered & gold) / max(1, len(gold))
            traces.append(
                {
                    "id": q["id"],
                    "query": q["query"],
                    "gold_doc_id": q["doc_id"],
                    "recall": recalls,
                    "top10": [
                        {
                            key: h["chunk"][key]
                            for key in ("id", "doc_id", "page", "kind", "evidence_keys")
                        }
                        for h in hits
                    ],
                }
            )
        return {
            "mode": mode,
            "rerank": rerank,
            "recall_at_3": float(np.mean([r["recall"][3] for r in traces])),
            "recall_at_5": float(np.mean([r["recall"][5] for r in traces])),
            "recall_at_10": float(np.mean([r["recall"][10] for r in traces])),
            "mean_query_ms": float(np.mean(times)),
            "p95_query_ms": float(np.percentile(times, 95)),
        }, traces

    for size in (256, 512, 1024):
        chunks = chunk_blocks(blocks, size, 50)
        for kind in ("flat", "hnsw", "ivf"):
            start = time.perf_counter()
            r = Retriever(chunks, index_type=kind)
            built = time.perf_counter() - start
            for mode in ("dense", "bm25", "hybrid"):
                metric, traces = evaluate(r, mode)
                report["experiments"].append(
                    {
                        "embedding": "tfidf",
                        "chunk_size": size,
                        "index": kind,
                        "chunks": len(chunks),
                        "build_seconds": built,
                        **metric,
                    }
                )
            if size == 512 and kind == "flat":
                r.save("artifacts/finqa-tfidf")
                Path("results/tfidf-hybrid-traces.json").write_text(
                    json.dumps(traces, indent=2)
                )
    if a.neural:
        chunks = chunk_blocks(blocks, 512, 50)
        start = time.perf_counter()
        r = Retriever(
            chunks,
            embedding="sbert:" + a.bge,
            reranker=a.reranker if a.flag_reranker else "ce:" + a.reranker,
        )
        built = time.perf_counter() - start
        for mode, rerank in [
            ("dense", False),
            ("bm25", False),
            ("hybrid", False),
            ("hybrid", True),
        ]:
            metric, traces = evaluate(r, mode, rerank)
            report["experiments"].append(
                {
                    "embedding": "BAAI/bge-small-en-v1.5",
                    "reranker_model": a.reranker if rerank else None,
                    "chunk_size": 512,
                    "index": "flat",
                    "chunks": len(chunks),
                    "build_seconds": built,
                    **metric,
                }
            )
            Path(f"results/bge-{mode}-rerank{rerank}-traces.json").write_text(
                json.dumps(traces, indent=2)
            )
        r.save("artifacts/finqa-bge")
    lookup = {d["doc_id"]: d for d in docs}
    calcs = []
    for q in queries:
        row = {
            "id": q["id"],
            "query": q["query"],
            "scope": "gold document supplied, no evidence/program labels supplied",
        }
        try:
            plan = propose(q["query"], lookup[q["doc_id"]])
            value = execute(docs, plan)
            expected = float(labels[q["id"]]["exe_ans"])
            pred = float(value["value"])
            correct = abs(pred - expected) <= max(0.01, abs(expected) * 0.001)
            row.update(
                answered=True, correct=correct, prediction=value, gold_answer=expected
            )
        except (CalculationError, ValueError, TypeError) as e:
            row.update(answered=False, reason=str(e))
        calcs.append(row)
    answered = [r for r in calcs if r["answered"]]
    report["numerical_planner"] = {
        "n": len(calcs),
        "answered": len(answered),
        "coverage": len(answered) / len(calcs),
        "execution_accuracy_all": sum(r.get("correct", False) for r in calcs)
        / len(calcs),
        "execution_accuracy_answered": sum(r["correct"] for r in answered)
        / len(answered)
        if answered
        else None,
        "scope": "document-scoped narrow rule planner; not global end-to-end QA",
    }
    report["limitations"] = [
        "Recall uses source evidence IDs, not answer accuracy",
        "Table atomic chunks can be long; neural encoder truncates at 512 tokens",
        "No test-based tuning; lexical refusal threshold is heuristic, not calibrated probability",
        "Planner coverage intentionally narrow; unsupported questions refused",
        "FinQA transcriptions are not full original PDFs",
    ]
    Path("results/finqa-benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    Path("results/numerical-cases.json").write_text(
        json.dumps(calcs, indent=2), encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    run()
