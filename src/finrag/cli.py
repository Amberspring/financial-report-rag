import argparse, json, time
from pathlib import Path
from .core import load_blocks, chunk_blocks, Retriever, answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/blocks.json")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--query")
    p.add_argument("--save", default="artifacts/index")
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text(encoding="utf-8"))
    start = time.perf_counter()
    r = Retriever(
        chunk_blocks(load_blocks(a.data), c["chunk_size"], c["overlap"]),
        c["embedding"],
        c["index"],
        c.get("reranker"),
    )
    r.save(a.save)
    print(
        json.dumps(
            {
                "chunks": len(r.chunks),
                "build_seconds": time.perf_counter() - start,
                "index_bytes": (Path(a.save) / "vectors.faiss").stat().st_size,
            },
            ensure_ascii=False,
        )
    )
    if a.query:
        print(
            json.dumps(
                answer(
                    r,
                    a.query,
                    threshold=c["refusal_threshold"],
                    rerank=bool(c.get("reranker")),
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
