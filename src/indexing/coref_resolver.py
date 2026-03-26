"""Coreference resolution using fastcoref on CPU.

NOTE: fastcoref chosen as CPU-feasible substitute for the theory's
spaCy+neuralcoref / AllenNLP coref. neuralcoref is abandoned (Python 3.6 era),
AllenNLP coref requires GPU for reasonable speed.
Reference: Otmazgin et al., 2023 — "F-COREF: Fast, Accurate and Easy to Use
Coreference Resolution"
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_coref_model = None


def _get_coref_model():
    """Load fastcoref model (cached)."""
    global _coref_model
    if _coref_model is None:
        from fastcoref import FCoref
        logger.info("Loading fastcoref model (CPU)...")
        _coref_model = FCoref(device="cpu")
        logger.info("fastcoref model loaded")
    return _coref_model


def resolve_coreferences(text: str, max_length: int = 4096) -> dict[str, Any]:
    """Run coreference resolution on a text.

    For long documents, processes in overlapping windows.

    Returns:
        {
            "clusters": {canonical_name: [alias1, alias2, ...]},
            "resolved_text": str  (text with pronouns replaced by entity names)
        }
    """
    model = _get_coref_model()

    # For short texts, process directly
    if len(text.split()) <= max_length:
        return _resolve_single(model, text)

    # For long texts, process in windows with overlap
    words = text.split()
    overlap = max_length // 4
    step = max_length - overlap
    all_clusters: dict[str, set[str]] = {}

    for i in range(0, len(words), step):
        window = " ".join(words[i:i + max_length])
        result = _resolve_single(model, window)
        for canonical, aliases in result["clusters"].items():
            key = canonical.lower().strip()
            if key not in all_clusters:
                all_clusters[key] = {canonical}
            all_clusters[key].update(aliases)

    # Convert sets to lists
    clusters = {
        list(aliases)[0]: list(aliases - {list(aliases)[0]})
        for aliases in all_clusters.values()
        if len(aliases) > 1
    }

    return {"clusters": clusters, "resolved_text": text}


def _resolve_single(model, text: str) -> dict[str, Any]:
    """Run coref on a single text segment."""
    try:
        preds = model.predict(texts=[text])
        clusters: dict[str, list[str]] = {}

        if preds and len(preds) > 0:
            pred = preds[0]
            # fastcoref returns clusters as lists of span tuples
            for cluster in pred.get_clusters(as_strings=True):
                if not cluster:
                    continue
                # Use the longest mention as canonical (likely the full name)
                canonical = max(cluster, key=len)
                aliases = [m for m in cluster if m != canonical]
                if aliases:
                    clusters[canonical] = aliases

        return {"clusters": clusters, "resolved_text": text}
    except Exception as e:
        logger.warning("Coref resolution failed: %s", e)
        return {"clusters": {}, "resolved_text": text}
