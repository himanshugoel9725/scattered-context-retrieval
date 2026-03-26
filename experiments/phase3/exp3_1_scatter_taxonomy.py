"""Exp 3.1: Scatter Pattern Taxonomy.

Classify 500 queries into 5 scatter types. Report per-category performance.
"""

import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np

from src.generation.llm_client import generate
from src.generation.prompts import SCATTER_CATEGORY_PROMPT
from src.retrieval.base import detect_query_entities
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)

SCATTER_TYPES = [
    "progressive_accumulation",
    "distributed_attributes",
    "contradictory_evolution",
    "cross_reference",
    "implicit",
]


def classify_scatter_type(query: str, entity: str | None = None,
                          num_chunks: int | str = "unknown",
                          positions: str = "unknown",
                          model: str = "gpt-4o-mini") -> str:
    """Classify a query's scatter pattern."""
    if not entity:
        # Try to detect entity from query text
        entities = detect_query_entities(query)
        entity = entities[0] if entities else "the subject"

    prompt = SCATTER_CATEGORY_PROMPT.format(
        entity=entity, question=query,
        num_chunks=num_chunks, positions=positions,
    )
    response = generate(prompt, model=model, temperature=0.0, max_tokens=50)
    response_lower = response.strip().lower()
    for st in SCATTER_TYPES:
        if st in response_lower:
            return st
    return "unknown"


def run(config: dict | None = None):
    if config is None:
        config = get_experiments_config()["phase3"]["exp3_1_scatter_taxonomy"]

    out_dir = results_dir("exp3_1")

    # Load pre-computed results from exp2_1 if available
    exp2_1_path = results_dir("exp2_1") / "exp2_1_results.json"
    if exp2_1_path.exists():
        with open(exp2_1_path) as f:
            exp2_data = json.load(f)
    else:
        logger.warning("Exp 2.1 results not found. Run exp2_1 first.")
        return

    classified = []
    for ds_name, entries in exp2_data.items():
        for entry in entries[:config.get("queries", 500) // max(len(exp2_data), 1)]:
            # Use detected entity from exp2_1 if available
            entity = entry.get("detected_entity")
            scatter_type = classify_scatter_type(entry["query"], entity=entity)
            entry["scatter_type"] = scatter_type
            entry["dataset"] = ds_name
            classified.append(entry)

    with open(out_dir / "exp3_1_results.json", "w") as f:
        json.dump(classified, f, indent=2, default=str)

    # Report distribution
    dist = Counter(e["scatter_type"] for e in classified)
    logger.info("Scatter type distribution: %s", dict(dist))

    # Plot per-category performance
    fig, ax = create_figure(figsize=(12, 5))
    strategies = ["BM25", "StandardSemantic", "EntityExpanded", "EntityFirst",
                  "Iterative", "HybridEntity"]
    x = np.arange(len(SCATTER_TYPES))
    width = 0.8 / max(len(strategies), 1)

    for i, strat in enumerate(strategies):
        means = []
        for st in SCATTER_TYPES:
            scores = [e[f"{strat}_rougeL"] for e in classified
                      if e.get("scatter_type") == st and f"{strat}_rougeL" in e]
            means.append(float(np.nanmean(scores)) if scores else 0)
        ax.bar(x + i * width, means, width, label=strat)

    ax.set_xticks(x + width * len(strategies) / 2)
    ax.set_xticklabels([t.replace("_", "\n") for t in SCATTER_TYPES], fontsize=8)
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Performance by Scatter Pattern Type")
    ax.legend()
    save_figure(fig, out_dir / "figure_scatter_taxonomy.pdf")

    return classified


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
