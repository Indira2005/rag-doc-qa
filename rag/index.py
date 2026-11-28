"""ChromaDB (dense) + BM25 (keyword) retrieval, fused with RRF, then cross-encoder reranking."""
import math
import re
from functools import lru_cache
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from rag import config
from rag.fusion import rrf_merge


@lru_cache
def embedder():
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache
def reranker():
    return CrossEncoder(config.RERANK_MODEL)


def _identity():
    from torch import nn
    return nn.Identity()  # ask for raw logits; we apply the sigmoid ourselves


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


class Index:
    def __init__(self, name: str = "docs", path: str | None = config.CHROMA_DIR):
        client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self.col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def add(self, chunks):
        if not chunks:
            raise ValueError("No text found in this PDF (scanned PDFs are not supported).")
        self.col.delete(where={"source": chunks[0]["source"]})  # re-upload replaces old chunks
        texts = [c["text"] for c in chunks]
        vectors = embedder().encode(texts, normalize_embeddings=True).tolist()
        for i in range(0, len(chunks), 1000):  # Chroma limits how much one call can add
            batch = chunks[i:i + 1000]
            self.col.add(
                ids=[c["id"] for c in batch],
                documents=texts[i:i + 1000],
                embeddings=vectors[i:i + 1000],
                metadatas=[{k: c[k] for k in ("source", "page", "section", "type")} for c in batch],
            )

    def delete(self, source):
        self.col.delete(where={"source": source})

    def _dense(self, query, n, where):
        vector = embedder().encode([query], normalize_embeddings=True).tolist()
        r = self.col.query(query_embeddings=vector, n_results=n, where=where)
        return [{"id": i, "text": t, **m}
                for i, t, m in zip(r["ids"][0], r["documents"][0], r["metadatas"][0])]

    def _keyword(self, query, n, where):
        # Rebuilt per query: simple and always in sync, fine for a few thousand chunks.
        data = self.col.get(where=where, include=["documents", "metadatas"])
        if not data["ids"]:
            return []
        scores = BM25Okapi([tokenize(t) for t in data["documents"]]).get_scores(tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [{"id": data["ids"][i], "text": data["documents"][i], **data["metadatas"][i]}
                for i in best if scores[i] > 0]

    def search(self, query, top_k=config.TOP_K, mode="rerank", source=None):
        """mode: dense | hybrid | rerank. `source` is the metadata filter."""
        where = {"source": source} if source else None
        n = max(config.CANDIDATES, top_k)
        hits = self._dense(query, n, where)
        if mode != "dense":
            hits = rrf_merge(hits, self._keyword(query, n, where))
        if mode == "rerank" and hits:
            logits = reranker().predict([(query, h["text"]) for h in hits],
                                        activation_fn=_identity())
            for hit, logit in zip(hits, logits):
                hit["score"] = 1 / (1 + math.exp(-float(logit)))  # absolute 0-1 score
            hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]

    def sources(self):
        return sorted({m["source"] for m in self.col.get(include=["metadatas"])["metadatas"]})
