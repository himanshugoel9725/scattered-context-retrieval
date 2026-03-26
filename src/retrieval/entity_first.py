"""Strategy C: Entity-First retrieval.

Detect query entity → retrieve ALL chunks for that entity from entity index →
rerank by cosine similarity to query → return top-K.
"""

import numpy as np

from src.indexing.chunker import ChunkMeta
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex, embed_texts
from src.retrieval.base import BaseRetriever, RetrievalResult, detect_query_entities


class EntityFirstRetriever(BaseRetriever):
    """Entity-first: get all entity chunks, then rerank semantically."""

    def __init__(self, vector_index: VectorIndex, entity_index: EntityIndex,
                 chunks: dict[str, ChunkMeta], fallback_semantic_k: int = 15):
        self.vector_index = vector_index
        self.entity_index = entity_index
        self.chunks = chunks
        self.fallback_k = fallback_semantic_k

    def retrieve(self, query: str, k: int = 15) -> list[RetrievalResult]:
        # Step 1: Detect entities in the query
        query_entities = detect_query_entities(query, entity_index=self.entity_index)

        # Step 2: Get ALL chunks for detected entities
        candidate_ids = set()
        for entity_name in query_entities:
            chunk_ids = self.entity_index.get_chunks_for_entity(entity_name)
            candidate_ids.update(chunk_ids)

        # Step 3: If no entities found or too few chunks, fall back to semantic
        if len(candidate_ids) < k // 2:
            semantic_hits = self.vector_index.search(query, k=k)
            for cid, _ in semantic_hits:
                candidate_ids.add(cid)

        # Step 4: Rerank all candidates by semantic similarity
        candidates = []
        for cid in candidate_ids:
            chunk = self.chunks.get(cid)
            if chunk:
                candidates.append((cid, chunk))

        if not candidates:
            # Ultimate fallback: pure semantic
            hits = self.vector_index.search(query, k=k)
            return [
                RetrievalResult(chunk_id=cid, text=self.chunks.get(cid, ChunkMeta("", "", "", 0, 0, 0, 0, 0)).text,
                                score=score, source="semantic_fallback")
                for cid, score in hits
            ]

        q_emb = embed_texts([query], show_progress=False)
        texts = [c.text for _, c in candidates]
        c_embs = embed_texts(texts, show_progress=False)
        scores = np.dot(c_embs, q_emb.T).flatten()

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        results = []
        for (cid, chunk), score in ranked[:k]:
            results.append(RetrievalResult(
                chunk_id=cid, text=chunk.text,
                score=float(score), source="entity_first",
            ))
        return results

    def name(self) -> str:
        return "EntityFirst"
