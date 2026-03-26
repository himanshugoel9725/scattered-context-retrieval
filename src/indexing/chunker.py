"""Fixed-size token chunker with overlap and positional metadata."""

from dataclasses import dataclass, field
from typing import Any

import tiktoken


@dataclass
class ChunkMeta:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    chunk_index: int       # 0-based index in document
    total_chunks: int      # total chunks in document
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def position_fraction(self) -> float:
        """Position as fraction of total chunks (0.0 = start, 1.0 = end)."""
        if self.total_chunks <= 1:
            return 0.0
        return self.chunk_index / (self.total_chunks - 1)

    @property
    def decile(self) -> int:
        """Which decile of the document this chunk falls in (0-9)."""
        return min(int(self.position_fraction * 10), 9)


class Chunker:
    """Token-based fixed-size chunker with configurable overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 128,
                 encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = tiktoken.get_encoding(encoding_name)

    def chunk_document(self, doc_id: str, text: str) -> list[ChunkMeta]:
        """Split a document into overlapping token chunks."""
        tokens = self.enc.encode(text)
        if not tokens:
            return []

        # Map token indices to character positions
        # We decode each token to figure out char boundaries
        char_offsets = []
        pos = 0
        for token_id in tokens:
            decoded = self.enc.decode([token_id])
            start = text.find(decoded, pos)
            if start == -1:
                start = pos  # fallback
            char_offsets.append(start)
            pos = start + len(decoded)

        chunks = []
        step = self.chunk_size - self.overlap
        i = 0
        while i < len(tokens):
            end = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:end]
            chunk_text = self.enc.decode(chunk_tokens)

            start_char = char_offsets[i]
            end_char = char_offsets[end - 1] + len(self.enc.decode([tokens[end - 1]])) if end > 0 else len(text)

            chunks.append(ChunkMeta(
                chunk_id=f"{doc_id}_c{len(chunks)}",
                doc_id=doc_id,
                text=chunk_text,
                start_char=start_char,
                end_char=min(end_char, len(text)),
                token_count=len(chunk_tokens),
                chunk_index=len(chunks),
                total_chunks=0,  # filled in below
                metadata={"doc_length": len(text)},
            ))

            if end >= len(tokens):
                break
            i += step

        # Fill total_chunks
        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks
