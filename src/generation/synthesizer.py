"""Multi-chunk synthesis: orders and formats scattered chunks for LLM generation."""

from dataclasses import dataclass
import random
from typing import Any

from src.generation.llm_client import generate
from src.generation.prompts import STANDARD_RAG_PROMPT, SCATTER_AWARE_SYNTHESIS_PROMPT


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] | None = None

    @property
    def position_fraction(self) -> float:
        """Position in document as 0-1 fraction (requires doc_length in metadata)."""
        if self.metadata and "doc_length" in self.metadata:
            return self.start_char / self.metadata["doc_length"]
        return 0.0


def order_chunks(chunks: list[Chunk], strategy: str, seed: int = 42) -> list[Chunk]:
    """Reorder chunks according to the given strategy."""
    if strategy == "chronological":
        return sorted(chunks, key=lambda c: c.start_char)
    elif strategy == "reverse_chronological":
        return sorted(chunks, key=lambda c: c.start_char, reverse=True)
    elif strategy == "relevance_ranked":
        # Assumes chunks are already ranked by relevance (retrieval order)
        return list(chunks)
    elif strategy == "entity_clustered":
        # Group by attribute type if available in metadata
        def attr_key(c: Chunk) -> str:
            return (c.metadata or {}).get("attribute_type", "unknown")
        return sorted(chunks, key=attr_key)
    elif strategy == "random":
        ordered = list(chunks)
        random.Random(seed).shuffle(ordered)
        return ordered
    else:
        raise ValueError(f"Unknown ordering strategy: {strategy}")


def to_synthesis_chunks(results: list, chunk_map: dict, doc_id: str = "") -> list[Chunk]:
    """Convert RetrievalResult objects to Chunk objects for synthesis."""
    chunks = []
    for r in results:
        meta = chunk_map.get(r.chunk_id)
        chunks.append(Chunk(
            chunk_id=r.chunk_id,
            doc_id=meta.doc_id if meta else doc_id,
            text=r.text,
            start_char=meta.start_char if meta else 0,
            end_char=meta.end_char if meta else 0,
        ))
    return chunks


def format_context(chunks: list[Chunk], numbered: bool = True) -> str:
    """Format chunks into a context string for the LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks):
        if numbered:
            parts.append(f"[Passage {i+1}]\n{chunk.text}")
        else:
            parts.append(chunk.text)
    return "\n\n".join(parts)


def synthesize(
    question: str,
    chunks: list[Chunk],
    entity: str | None = None,
    ordering: str = "chronological",
    model: str | None = None,
    provider: str | None = None,
    scatter_aware: bool = True,
) -> dict[str, Any]:
    """Generate an answer from scattered chunks.

    Returns dict with: answer, ordering_used, num_chunks, model
    """
    ordered = order_chunks(chunks, ordering)
    context = format_context(ordered)

    if scatter_aware and entity:
        prompt = SCATTER_AWARE_SYNTHESIS_PROMPT.format(
            entity=entity, context=context, question=question
        )
    else:
        prompt = STANDARD_RAG_PROMPT.format(
            context=context, question=question
        )

    answer = generate(prompt, model=model, provider=provider)

    return {
        "answer": answer,
        "ordering_used": ordering,
        "num_chunks": len(chunks),
        "prompt_length": len(prompt),
    }
