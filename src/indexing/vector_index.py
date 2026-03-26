"""FAISS CPU vector index for chunk embeddings."""

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.utils.config import data_dir, get_models_config

logger = logging.getLogger(__name__)

_model_cache: dict[str, Any] = {}


def _get_embedding_model(model_name: str | None = None):
    """Load a sentence-transformers model (cached in memory)."""
    if model_name is None:
        config = get_models_config()
        model_name = config["embeddings"]["local"]["dev"]["name"]

    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str | None = None,
                batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Embed a list of texts into vectors."""
    model = _get_embedding_model(model_name)
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress,
        normalize_embeddings=True
    )
    return np.array(embeddings, dtype=np.float32)


class VectorIndex:
    """FAISS-based vector index for chunk retrieval."""

    def __init__(self, dim: int | None = None, index_type: str = "flat"):
        config = get_models_config()
        self.dim = dim or config["embeddings"]["local"]["dev"]["dim"]
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dim)  # inner product (cosine for normalized)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []

    def add(self, chunk_ids: list[str], texts: list[str],
            embeddings: np.ndarray | None = None, model_name: str | None = None):
        """Add chunks to the index."""
        if embeddings is None:
            embeddings = embed_texts(texts, model_name=model_name)
        assert embeddings.shape[0] == len(chunk_ids)
        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)
        self.texts.extend(texts)

    def search(self, query: str, k: int = 5,
               model_name: str | None = None) -> list[tuple[str, float]]:
        """Search for top-k similar chunks. Returns (chunk_id, score) pairs."""
        q_emb = embed_texts([query], model_name=model_name, show_progress=False)
        scores, indices = self.index.search(q_emb, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results

    def search_by_embedding(self, embedding: np.ndarray,
                            k: int = 5) -> list[tuple[str, float]]:
        """Search by pre-computed embedding vector."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        scores, indices = self.index.search(embedding, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results

    def save(self, name: str):
        """Save index to disk."""
        out_dir = data_dir("vector_indices")
        faiss.write_index(self.index, str(out_dir / f"{name}.faiss"))
        with open(out_dir / f"{name}.meta.pkl", "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "texts": self.texts, "dim": self.dim}, f)
        logger.info("Saved vector index '%s' (%d vectors)", name, self.index.ntotal)

    @classmethod
    def load(cls, name: str) -> "VectorIndex":
        """Load index from disk."""
        idx_dir = data_dir("vector_indices")
        index = faiss.read_index(str(idx_dir / f"{name}.faiss"))
        with open(idx_dir / f"{name}.meta.pkl", "rb") as f:
            meta = pickle.load(f)

        vi = cls.__new__(cls)
        vi.index = index
        vi.chunk_ids = meta["chunk_ids"]
        vi.texts = meta["texts"]
        vi.dim = meta["dim"]
        logger.info("Loaded vector index '%s' (%d vectors)", name, index.ntotal)
        return vi
