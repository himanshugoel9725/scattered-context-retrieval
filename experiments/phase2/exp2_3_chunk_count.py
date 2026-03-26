"""Exp 2.3: Chunk Count Analysis.

Strategy E, vary K from 1 to 30. Split by scatter factor bins.
Generate Figure 10.
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
from src.retrieval.hybrid_entity import HybridEntityRetriever
from src.generation.synthesizer import synthesize, to_synthesis_chunks
from src.retrieval.base import detect_query_entities
from src.evaluation.standard_metrics import compute_rouge_l
from src.evaluation.metrics import compute_scatter_factor
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)


def run(config: dict | None = None):
    if config is None:
        config = get_experiments_config()["phase2"]["exp2_3_chunk_count"]

    out_dir = results_dir("exp2_3")
    chunker = Chunker(chunk_size=512, overlap=128)
    k_values = config.get("k_values", [1, 3, 5, 7, 10, 15, 20, 30])

    datasets = ["quality"]
    all_results = []

    for ds_name in datasets:
        dataset = load_dataset(ds_name)
        if not dataset:
            continue

        for doc in dataset[:config.get("max_docs", 30)]:
            text = clean_text(doc.text)
            chunks = chunker.chunk_document(doc.doc_id, text)
            if len(chunks) < 10:
                continue
            chunk_map = {c.chunk_id: c for c in chunks}

            vector_idx = VectorIndex()
            vector_idx.add([c.chunk_id for c in chunks], [c.text for c in chunks])
            entity_idx = EntityIndex(doc.doc_id)
            entity_idx.build_from_chunks(chunks)

            retriever = HybridEntityRetriever(vector_idx, entity_idx, chunk_map)

            for qa in doc.questions[:5]:
                # Compute SF for most-mentioned entity in query context
                top_ents = entity_idx.get_top_entities(1)
                sf = compute_scatter_factor(
                    top_ents[0]["canonical"], top_ents[0]["chunk_ids"], chunk_map
                ) if top_ents else 0.0
                sf_bin = "low" if sf < 0.1 else ("medium" if sf < 0.3 else "high")

                # Detect entity for scatter-aware synthesis
                entities = detect_query_entities(qa["question"], entity_idx)
                entity = entities[0] if entities else None

                entry = {"query": qa["question"], "sf": sf, "sf_bin": sf_bin}
                for k in k_values:
                    retrieved = retriever.retrieve(qa["question"], k=k)
                    context_texts = [r.text for r in retrieved]
                    syn_chunks = to_synthesis_chunks(retrieved, chunk_map, doc.doc_id)
                    result = synthesize(qa["question"], syn_chunks, entity=entity, model="gpt-4o-mini")
                    answer = result["answer"]
                    rouge = compute_rouge_l(answer, qa.get("answer", ""))
                    entry[f"rougeL_k{k}"] = rouge["rougeL_fmeasure"]

                all_results.append(entry)

    with open(out_dir / "exp2_3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    _plot_figure10(all_results, k_values, out_dir)
    logger.info("Exp 2.3 complete. Results in %s", out_dir)
    return all_results


def _plot_figure10(results: list, k_values: list, out_dir: Path):
    """Figure 10: ROUGE-L vs K, split by SF bin."""
    fig, ax = create_figure(figsize=(8, 5))
    for sf_bin in ["low", "medium", "high"]:
        subset = [r for r in results if r["sf_bin"] == sf_bin]
        if not subset:
            continue
        means = [float(np.nanmean([r[f"rougeL_k{k}"] for r in subset if f"rougeL_k{k}" in r]))
                 for k in k_values]
        ax.plot(k_values, means, marker="o", label=f"SF={sf_bin}")

    ax.set_xlabel("K (retrieved chunks)")
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Answer Quality vs. Number of Retrieved Chunks")
    ax.legend()
    save_figure(fig, out_dir / "figure10_chunk_count.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
