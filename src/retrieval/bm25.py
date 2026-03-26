"""BM25 baseline retrieval."""

from rank_bm25 import BM25Okapi

from src.indexing.chunker import ChunkMeta
from src.retrieval.base import BaseRetriever, RetrievalResult


class BM25Retriever(BaseRetriever):
    """BM25-based keyword retrieval."""

    def __init__(self, chunks: list[ChunkMeta]):
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 15) -> list[RetrievalResult]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[-k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            c = self.chunks[idx]
            results.append(RetrievalResult(
                chunk_id=c.chunk_id, text=c.text,
                score=float(scores[idx]), source="bm25",
            ))
        return results

    def name(self) -> str:
        return "BM25"
