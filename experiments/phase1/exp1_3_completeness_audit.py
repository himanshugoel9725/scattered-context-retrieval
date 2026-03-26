"""Exp 1.3: Chunk Retrieval Completeness Audit.

For scattered queries, use entity index as gold for relevant chunks.
Compute Retrieval Completeness@K at different K values and SF bins.
Generate Figure 5.
"""

import json
import logging
from pathlib import Path

import numpy as np

from src.data.loaders import load_dataset
from src.data.processors import clean_text
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex
from src.retrieval.standard_semantic import StandardSemanticRetriever
from src.evaluation.metrics import compute_scatter_factor, retrieval_completeness_at_k
from src.evaluation.gold_schema import GoldEvidence
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)


def run(config: dict | None = None):
    """Run Experiment 1.3: Completeness Audit."""
    if config is None:
        config = get_experiments_config()["phase1"]["exp1_3_completeness_audit"]

    out_dir = results_dir("exp1_3")
    chunker = Chunker(chunk_size=512, overlap=128)
    k_values = config.get("k_values", [1, 3, 5, 7, 10, 15, 20])
    n_queries = config.get("queries", 200)

    # Use multiple datasets for sufficient document length
    all_docs = []
    for ds_name in ["qasper", "cuad", "narrativeqa"]:
        ds = load_dataset(ds_name)
        if ds:
            all_docs.extend(ds[:config.get("max_docs", 50)])

    if not all_docs:
        logger.error("Could not load any datasets")
        return

    results = []  # list of {entity, sf, sf_bin, k: RC@K, ...}

    for doc in all_docs:
        text = clean_text(doc.text)
        chunks = chunker.chunk_document(doc.doc_id, text)
        chunk_map = {c.chunk_id: c for c in chunks}

        if len(chunks) < 5:
            continue

        entity_idx = EntityIndex(doc.doc_id)
        entity_idx.build_from_chunks(chunks)

        vector_idx = VectorIndex()
        vector_idx.add([c.chunk_id for c in chunks], [c.text for c in chunks])
        semantic = StandardSemanticRetriever(vector_idx, chunk_map)

        for ent_data in entity_idx.get_top_entities(5):
            sf = compute_scatter_factor(ent_data["canonical"], ent_data["chunk_ids"], chunk_map)
            gold = GoldEvidence(
                query_id=f"{doc.doc_id}:{ent_data['canonical']}",
                entity_id=ent_data["canonical"],
                gold_chunk_ids=ent_data["chunk_ids"],
            )

            query = f"Describe everything about {ent_data['canonical']}"
            record = {
                "entity": ent_data["canonical"],
                "scatter_factor": sf,
                "sf_bin": "low" if sf < 0.1 else ("medium" if sf < 0.3 else "high"),
                "n_gold_chunks": len(gold.gold_chunk_ids),
            }

            for k in k_values:
                retrieved = semantic.retrieve(query, k=k)
                rc = retrieval_completeness_at_k([r.chunk_id for r in retrieved], gold)
                record[f"rc_at_{k}"] = rc

            results.append(record)
            if len(results) >= n_queries:
                break
        if len(results) >= n_queries:
            break

    with open(out_dir / "exp1_3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot_figure5(results, k_values, out_dir)
    logger.info("Exp 1.3 complete. %d entries. Results in %s", len(results), out_dir)
    return results


def _plot_figure5(results: list, k_values: list, out_dir: Path):
    """Figure 5: RC@K at different K, split by SF bins."""
    fig, ax = create_figure(figsize=(8, 5))
    bins = ["low", "medium", "high"]
    for sf_bin in bins:
        subset = [r for r in results if r["sf_bin"] == sf_bin]
        if not subset:
            continue
        means = [np.mean([r[f"rc_at_{k}"] for r in subset]) for k in k_values]
        ax.plot(k_values, means, marker="o", label=f"SF={sf_bin}")

    ax.set_xlabel("K (retrieved chunks)")
    ax.set_ylabel("Retrieval Completeness@K")
    ax.set_title("Retrieval Completeness by Scatter Factor Bin")
    ax.legend()
    save_figure(fig, out_dir / "figure5_completeness_by_sf.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
