"""Strategy A: Standard semantic (top-K cosine) retrieval."""

from src.indexing.chunker import ChunkMeta
from src.indexing.vector_index import VectorIndex
from src.retrieval.base import BaseRetriever, RetrievalResult


class StandardSemanticRetriever(BaseRetriever):
    """Top-K by cosine similarity from FAISS vector index."""

    def __init__(self, vector_index: VectorIndex, chunks: dict[str, ChunkMeta]):
        self.vector_index = vector_index
        self.chunks = chunks  # chunk_id -> ChunkMeta

    def retrieve(self, query: str, k: int = 15) -> list[RetrievalResult]:
        hits = self.vector_index.search(query, k=k)
        results = []
        for chunk_id, score in hits:
            chunk = self.chunks.get(chunk_id)
            text = chunk.text if chunk else ""
            results.append(RetrievalResult(
                chunk_id=chunk_id, text=text,
                score=score, source="semantic",
            ))
        return results

    def name(self) -> str:
        return "StandardSemantic"
