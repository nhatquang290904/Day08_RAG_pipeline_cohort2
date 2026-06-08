"""
Task 5 - Semantic search over the local vector index from Task 4.
"""

try:
    from src.task4_chunking_indexing import embed_query, load_index
except ModuleNotFoundError:
    from task4_chunking_indexing import embed_query, load_index


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by vector similarity.

    Args:
        query: User query.
        top_k: Maximum number of results.

    Returns:
        List of {"content": str, "score": float, "metadata": dict}
        sorted by score descending.
    """
    if top_k <= 0:
        return []

    query_embedding = embed_query(query)
    chunks = load_index()

    results: list[dict] = []
    for chunk in chunks:
        score = _cosine_similarity(query_embedding, chunk.get("embedding", []))
        results.append(
            {
                "content": chunk["content"],
                "score": score,
                "metadata": chunk.get("metadata", {}),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5):
        preview = result["content"][:100].encode("ascii", errors="replace").decode("ascii")
        print(f"[{result['score']:.3f}] {preview}...")
