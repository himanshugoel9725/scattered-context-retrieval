"""Strategy D: Iterative entity discovery retrieval.

Retrieve → extract entities → retrieve more about those entities →
repeat for N rounds. Deduplicate across rounds.
"""

from src.indexing.chunker import ChunkMeta
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex
from src.retrieval.base import BaseRetriever, RetrievalResult, detect_query_entities


class IterativeRetriever(BaseRetriever):
    """Multi-round retrieval with entity discovery at each step."""

    def __init__(self, vector_index: VectorIndex, entity_index: EntityIndex,
                 chunks: dict[str, ChunkMeta], rounds: int = 2):
        self.vector_index = vector_index
        self.entity_index = entity_index
        self.chunks = chunks
        self.rounds = rounds

    def retrieve(self, query: str, k: int = 15) -> list[RetrievalResult]:
        all_results: dict[str, RetrievalResult] = {}
        known_entities: set[str] = set()
        per_round_k = max(k // self.rounds, 5)

        for round_num in range(self.rounds):
            if round_num == 0:
                # First round: semantic retrieval
                hits = self.vector_index.search(query, k=per_round_k)
                for chunk_id, score in hits:
                    if chunk_id not in all_results:
                        chunk = self.chunks.get(chunk_id)
                        all_results[chunk_id] = RetrievalResult(
                            chunk_id=chunk_id, text=chunk.text if chunk else "",
                            score=score, source=f"semantic_round{round_num}",
                        )
            else:
                # Subsequent rounds: entity-expanded retrieval
                for entity_name in new_entities:
                    entity_chunks = self.entity_index.get_chunks_for_entity(entity_name)
                    for cid in entity_chunks:
                        if cid not in all_results:
                            chunk = self.chunks.get(cid)
                            if chunk:
                                # Score by how many discovered entities appear in this chunk
                                ents_in_chunk = set(self.entity_index.get_entities_in_chunk(cid))
                                overlap = len(ents_in_chunk & known_entities)
                                all_results[cid] = RetrievalResult(
                                    chunk_id=cid, text=chunk.text,
                                    score=0.5 + 0.1 * overlap,
                                    source=f"entity_round{round_num}",
                                )

            # Discover entities in current results
            new_entities = set()
            for r in all_results.values():
                ents = self.entity_index.get_entities_in_chunk(r.chunk_id)
                for e in ents:
                    if e not in known_entities:
                        new_entities.add(e)
                        known_entities.add(e)

            # Also detect entities from query (first round)
            if round_num == 0:
                query_ents = detect_query_entities(query, entity_index=self.entity_index)
                for e in query_ents:
                    known_entities.add(e.lower())
                    new_entities.add(e.lower())

            if not new_entities:
                break  # No new entities discovered

        # Return top-K by score
        sorted_results = sorted(all_results.values(), key=lambda r: r.score, reverse=True)
        return sorted_results[:k]

    def name(self) -> str:
        return f"Iterative(rounds={self.rounds})"
