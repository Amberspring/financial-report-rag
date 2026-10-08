import json, time, hashlib, platform
import faiss

faiss.omp_set_num_threads(1)
from pathlib import Path
from finrag.core import load_blocks, chunk_blocks, Retriever, answer

blocks = load_blocks("data/blocks.json")
qa = json.loads(Path("data/qa.json").read_text(encoding="utf-8"))
results = []
details = []
for size, overlap in [(256, 50), (512, 100), (1024, 200)]:
    chunks = chunk_blocks(blocks, size, overlap)
    for index_type in ("flat", "ivf", "hnsw"):
        start = time.perf_counter()
        r = Retriever(chunks, index_type=index_type)
        build = time.perf_counter() - start
        for mode in ("dense", "bm25", "hybrid"):
            recalls = {k: [] for k in (3, 5, 10)}
            evidence = {k: [] for k in (3, 5, 10)}
            lat = []
            refusals = []
            for item in qa:
                start = time.perf_counter()
                hits = r.search(item["query"], 10, mode)
                lat.append((time.perf_counter() - start) * 1000)
                if item["doc_ids"]:
                    for k in recalls:
                        found = {h["chunk"]["doc_id"] for h in hits[:k]}
                        recalls[k].append(
                            len(found & set(item["doc_ids"])) / len(item["doc_ids"])
                        )
                        gold = item["evidence"]
                        covered = sum(
                            any(
                                h["chunk"]["doc_id"] == g["doc_id"]
                                and g["contains"] in h["chunk"]["text"]
                                for h in hits[:k]
                            )
                            for g in gold
                        )
                        evidence[k].append(covered / len(gold))
                ans = answer(r, item["query"], mode)
                refusals.append(ans["refused"] == (not item["doc_ids"]))
                if size == 512 and index_type == "flat":
                    details.append(
                        {"query": item["query"], "mode": mode, "result": ans}
                    )
            results.append(
                dict(
                    chunk_size_chars=size,
                    overlap_chars=overlap,
                    index=index_type,
                    mode=mode,
                    chunks=len(chunks),
                    build_seconds=build,
                    index_bytes=len(__import__("faiss").serialize_index(r.index)),
                    mean_query_ms=sum(lat) / len(lat),
                    refusal_fixture_pass_rate=sum(refusals) / len(refusals),
                    **{
                        f"document_recall_at_{k}": sum(v) / len(v)
                        for k, v in recalls.items()
                    },
                    **{
                        f"evidence_recall_at_{k}": sum(v) / len(v)
                        for k, v in evidence.items()
                    },
                )
            )
out = {
    "status": "measured",
    "scope": "authored_synthetic_fixture",
    "embedding": "char_tfidf_not_BGE",
    "reranker": "not_run",
    "qa_count": len(qa),
    "answerable_qa_count": sum(bool(x["doc_ids"]) for x in qa),
    "hardware": platform.platform(),
    "data_sha256": hashlib.sha256(Path("data/blocks.json").read_bytes()).hexdigest(),
    "qa_sha256": hashlib.sha256(Path("data/qa.json").read_bytes()).hexdigest(),
    "runs": results,
    "limitations": "Tiny corpus; document-level recall; no real-report or model-generation accuracy claim.",
}
Path("results").mkdir(exist_ok=True)
Path("results/retrieval.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
Path("results/cases.json").write_text(
    json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(out, ensure_ascii=False))
