from typing import Any, Dict, List


def rrf_merge(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Merge vector and BM25 ranked lists using Reciprocal Rank Fusion.

    RRF formula:  score(d) = Σ  1 / (k + rank(d, list))
    where rank is 1-based and k=60 is the standard smoothing constant.

    Args:
        vector_results: Chunks from ChromaDB, ordered by descending similarity.
        bm25_results:   Chunks from BM25, ordered by descending BM25 score.
        k:              RRF smoothing constant (default 60).

    Returns:
        Deduplicated list of chunks sorted by descending RRF score, each chunk
        carrying an ``rrf_score`` field.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_store: Dict[str, Dict[str, Any]] = {}

    def _key(chunk: Dict[str, Any]) -> str:
        """Stable deduplication key — prefer explicit IDs, fall back to content hash."""
        for field in ("id", "chunk_id"):
            if chunk.get(field):
                return str(chunk[field])
        text = str(chunk.get("text", ""))
        return "|".join([
            str(chunk.get("document_id", "")),
            str(chunk.get("page", "")),
            text[:200],
        ])

    def _accumulate(results: List[Dict[str, Any]]) -> None:
        for rank, chunk in enumerate(results, start=1):
            key = _key(chunk)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in chunk_store or chunk.get("score", 0) > chunk_store[key].get("score", 0):
                chunk_store[key] = chunk

    _accumulate(vector_results)
    _accumulate(bm25_results)

    merged = []
    for key, rrf_score in sorted(rrf_scores.items(), key=lambda t: t[1], reverse=True):
        chunk = chunk_store[key].copy()
        chunk["rrf_score"] = round(rrf_score, 6)
        merged.append(chunk)

    return merged
