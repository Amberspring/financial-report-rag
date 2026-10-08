from dataclasses import dataclass, asdict
from pathlib import Path
import re, json, math, hashlib
from collections import Counter
import numpy as np


@dataclass
class Block:
    doc_id: str
    page: int
    text: str
    kind: str = "text"
    bbox: tuple | None = None
    evidence_keys: tuple = ()


@dataclass
class Chunk:
    id: str
    doc_id: str
    page: int
    text: str
    kind: str
    bbox: tuple | None = None
    evidence_keys: tuple = ()


def tokenize(s):
    s = s.lower()
    runs = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", s)
    out = []
    for r in runs:
        if "\u4e00" <= r[0] <= "\u9fff":
            out.extend(r)
            out.extend(r[i : i + 2] for i in range(len(r) - 1))
        else:
            out.append(r)
    return out


def evidence_overlap(query, text):
    # Single Chinese character matches are too weak for a refusal decision.
    q = set(t for t in tokenize(query) if len(t) > 1)
    if not q:
        q = set(tokenize(query))
    evidence = set(tokenize(text))
    years = set(re.findall(r"20\d{2}", query))
    if years and not years <= set(re.findall(r"20\d{2}", text)):
        return 0.0
    return len(q & evidence) / max(1, len(q))


def chunk_blocks(blocks, size=512, overlap=50):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    out = []
    for b in blocks:
        # A detected table is atomic, even when larger than the target size.
        starts = [0] if b.kind == "table" else range(0, len(b.text), size - overlap)
        for start in starts:
            text = b.text if b.kind == "table" else b.text[start : start + size]
            if not text.strip():
                continue
            key = f"{b.doc_id}:{b.page}:{b.kind}:{b.bbox}:{start}:{text}"
            out.append(
                Chunk(
                    hashlib.sha256(key.encode()).hexdigest()[:16],
                    b.doc_id,
                    b.page,
                    text,
                    b.kind,
                    b.bbox,
                    b.evidence_keys,
                )
            )
            if b.kind == "table" or start + size >= len(b.text):
                break
    return out


def parse_pdf(path, doc_id=None):
    import pdfplumber

    blocks = []
    doc_id = doc_id or Path(path).stem
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = page.find_tables()
            words = page.extract_words()
            excluded = [t.bbox for t in tables]

            def in_table(w):
                x = (w["x0"] + w["x1"]) / 2
                y = (w["top"] + w["bottom"]) / 2
                return any(
                    x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in excluded
                )

            words = [w for w in words if not in_table(w)]
            # Detect a central gutter only when no non-table word crosses it.
            mid = page.width / 2
            columns = [words]
            if words and not any(
                w["x0"] < mid - 8 and w["x1"] > mid + 8 for w in words
            ):
                left = [w for w in words if w["x1"] <= mid]
                right = [w for w in words if w["x0"] >= mid]
                if len(left) > 3 and len(right) > 3:
                    columns = [left, right]
            for col in columns:
                lines = {}
                for w in col:
                    lines.setdefault(round(w["top"] / 3) * 3, []).append(w)
                text = "\n".join(
                    " ".join(w["text"] for w in sorted(ws, key=lambda w: w["x0"]))
                    for _, ws in sorted(lines.items())
                )
                if text.strip():
                    blocks.append(Block(doc_id, page.page_number, text))
            for table in tables:
                matrix = table.extract()
                text = "\n".join(
                    " | ".join(str(v or "").replace("\n", " ") for v in row)
                    for row in matrix
                )
                blocks.append(
                    Block(doc_id, page.page_number, text, "table", tuple(table.bbox))
                )
    if not blocks:
        raise ValueError(
            "No extractable text: scanned PDF requires OCR; it is not silently accepted."
        )
    return blocks


class BM25:
    def __init__(self, texts):
        self.rows = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(r.values()) for r in self.rows])
        self.avg = max(float(self.lengths.mean()), 1)
        df = Counter(t for r in self.rows for t in r)
        n = len(texts)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, q):
        scores = np.zeros(len(self.rows))
        for t in set(tokenize(q)):
            tf = np.array([r[t] for r in self.rows])
            scores += (
                self.idf.get(t, 0)
                * tf
                * 2.5
                / (tf + 1.5 * (0.25 + 0.75 * self.lengths / self.avg))
            )
        return scores


class Retriever:
    def __init__(self, chunks, embedding="tfidf", index_type="flat", reranker=None):
        if not chunks:
            raise ValueError("Empty corpus")
        import faiss

        self.chunks = chunks
        self.embedding = embedding
        self.reranker = None
        texts = [c.text for c in chunks]
        self.bm25 = BM25(texts)
        if embedding == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.encoder = TfidfVectorizer(
                analyzer="char", ngram_range=(1, 3), max_features=4096
            )
            vectors = self.encoder.fit_transform(texts).toarray().astype("float32")
        elif embedding.startswith("sbert:"):
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(embedding[6:], device="cpu")
            self.encoder.max_seq_length = 512
            vectors = self.encoder.encode(
                texts, batch_size=16, convert_to_numpy=True
            ).astype("float32")
        else:
            from FlagEmbedding import FlagModel

            self.encoder = FlagModel(
                embedding,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=False,
            )
            vectors = np.asarray(self.encoder.encode(texts), dtype="float32")
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 64
        elif index_type == "ivf":
            nlist = max(1, min(32, len(chunks) // 40))
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(vectors)
            self.index.nprobe = min(nlist, 8)
        else:
            raise ValueError("Unknown index_type")
        self.index.add(vectors)
        if reranker and reranker.startswith("ce:"):
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(reranker[3:], device="cpu", max_length=512)
        elif reranker:
            from FlagEmbedding import FlagReranker

            self.reranker = FlagReranker(reranker, use_fp16=False)

    def vector(self, q):
        import faiss

        v = (
            self.encoder.transform([q]).toarray()
            if self.embedding == "tfidf"
            else self.encoder.encode(
                ["Represent this sentence for searching relevant passages: " + q],
                convert_to_numpy=True,
            )
            if self.embedding.startswith("sbert:")
            else self.encoder.encode_queries([q])
        )
        v = np.asarray(v, dtype="float32")
        faiss.normalize_L2(v)
        return v

    def search(self, q, k=5, mode="hybrid", rerank=False):
        if mode not in ("dense", "bm25", "hybrid"):
            raise ValueError("Unknown retrieval mode")
        n = len(self.chunks)
        candidate = min(n, max(20, k))
        dense = []
        if mode != "bm25":
            v = self.vector(q)
            ds, di = self.index.search(v, candidate)
            dense = [(int(i), float(s)) for i, s in zip(di[0], ds[0]) if i >= 0]
        bs = self.bm25.scores(q)
        sparse = [
            (int(i), float(bs[i]))
            for i in np.argsort(-bs, kind="stable")[:candidate]
            if bs[i] > 0
        ]
        if mode == "dense":
            rank = dense
        elif mode == "bm25":
            rank = sparse
        else:
            fused = {}
            for route in (dense, sparse):
                for r, (i, _) in enumerate(route):
                    fused[i] = fused.get(i, 0) + 1 / (60 + r + 1)
            rank = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
        if rerank:
            if self.reranker is None:
                raise ValueError("No BGE reranker configured")
            scores = np.atleast_1d(
                self.reranker.predict(
                    [[q, self.chunks[i].text] for i, _ in rank], batch_size=16
                )
                if hasattr(self.reranker, "predict")
                else self.reranker.compute_score(
                    [[q, self.chunks[i].text] for i, _ in rank], normalize=True
                )
            )
            rank = sorted(
                [(i, float(s)) for (i, _), s in zip(rank, scores)], key=lambda x: -x[1]
            )
        dense_scores = dict(dense)
        return [
            {
                "chunk": asdict(self.chunks[i]),
                "score": s,
                "cosine": dense_scores.get(i, 0),
                "lexical_overlap": evidence_overlap(q, self.chunks[i].text),
            }
            for i, s in rank[:k]
        ]

    def save(self, directory):
        import faiss, joblib

        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "vectors.faiss"))
        (p / "metadata.json").write_text(
            json.dumps(
                {
                    "embedding": self.embedding,
                    "chunks": [asdict(c) for c in self.chunks],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.embedding == "tfidf":
            joblib.dump(self.encoder, p / "encoder.joblib")

    @classmethod
    def load_trusted(cls, directory):
        """Only load artifacts created by you; joblib and FAISS files are not safe untrusted inputs."""
        import faiss, joblib

        p = Path(directory)
        meta = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj.chunks = [Chunk(**c) for c in meta["chunks"]]
        obj.embedding = meta["embedding"]
        obj.reranker = None
        obj.index = faiss.read_index(str(p / "vectors.faiss"))
        obj.bm25 = BM25([c.text for c in obj.chunks])
        if obj.embedding == "tfidf":
            obj.encoder = joblib.load(p / "encoder.joblib")
        elif obj.embedding.startswith("sbert:"):
            from sentence_transformers import SentenceTransformer

            obj.encoder = SentenceTransformer(obj.embedding[6:], device="cpu")
        else:
            from FlagEmbedding import FlagModel

            obj.encoder = FlagModel(
                obj.embedding,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=False,
            )
        return obj


def answer(retriever, q, mode="hybrid", threshold=0.2, rerank=False):
    hits = retriever.search(q, 5, mode, rerank)
    # A lexical gate for the CPU baseline, not a calibrated probability.
    supported = [h for h in hits if h["lexical_overlap"] >= threshold]
    if not supported:
        return {
            "answer": "资料不足，无法确定。",
            "refused": True,
            "citations": [],
            "hits": hits,
            "mode": "extractive",
        }
    selected = supported[:2]
    citations = []
    parts = []
    for i, h in enumerate(selected, 1):
        c = h["chunk"]
        parts.append(f"{c['text']} [{i}]")
        citations.append(
            {
                "id": i,
                "doc_id": c["doc_id"],
                "page": c["page"],
                "chunk_id": c["id"],
                "quote": c["text"],
                "bbox": c["bbox"],
            }
        )
    return {
        "answer": "\n".join(parts),
        "refused": False,
        "citations": citations,
        "hits": hits,
        "mode": "extractive",
    }


def load_blocks(path):
    return [Block(**r) for r in json.loads(Path(path).read_text(encoding="utf-8"))]
