"""Exp 2.5: Ordering Strategy Comparison.

5 orderings × 2 domains (narrative + legal) × 200 queries.
Random ordering repeated 3 times and averaged.
Generate Figure 12.
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
from src.generation.synthesizer import to_synthesis_chunks, order_chunks, format_context, synthesize
from src.generation.llm_client import generate
from src.generation.prompts import SCATTER_AWARE_SYNTHESIS_PROMPT
from src.retrieval.base import detect_query_entities
from src.evaluation.standard_metrics import compute_rouge_l
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)

ORDERINGS = ["chronological", "reverse_chronological", "relevance_ranked", "entity_clustered", "random"]


def run(config: dict | None = None):
    if config is None:
        config = get_experiments_config()["phase2"]["exp2_5_ordering"]

    out_dir = results_dir("exp2_5")
    chunker = Chunker(chunk_size=512, overlap=128)
    n_queries = config.get("queries_per_domain", 200)

    results = {}

    for ds_name in ["quality", "cuad"]:
        dataset = load_dataset(ds_name)
        if not dataset:
            continue

        ds_results = {o: [] for o in ORDERINGS}
        queries_done = 0

        for doc in dataset:
            if queries_done >= n_queries:
                break
            text = clean_text(doc.text)
            chunks = chunker.chunk_document(doc.doc_id, text)
            if len(chunks) < 5:
                continue
            chunk_map = {c.chunk_id: c for c in chunks}

            vector_idx = VectorIndex()
            vector_idx.add([c.chunk_id for c in chunks], [c.text for c in chunks])
            entity_idx = EntityIndex(doc.doc_id)
            entity_idx.build_from_chunks(chunks)

            retriever = HybridEntityRetriever(vector_idx, entity_idx, chunk_map)

            for qa in doc.questions:
                if queries_done >= n_queries:
                    break
                retrieved = retriever.retrieve(qa["question"], k=15)

                # Convert to Chunk objects for ordering
                syn_chunks = to_synthesis_chunks(retrieved, chunk_map, doc.doc_id)

                # Detect entity for scatter-aware prompt
                entities = detect_query_entities(qa["question"], entity_idx)
                entity = entities[0] if entities else "the subject"

                for ordering in ORDERINGS:
                    if ordering == "random":
                        # Average over 3 random orderings with global seeds
                        rouge_scores = []
                        for seed in [42, 123, 456]:
                            ordered = order_chunks(syn_chunks, ordering, seed=seed)
                            context = format_context(ordered)
                            prompt = SCATTER_AWARE_SYNTHESIS_PROMPT.format(
                                context=context, question=qa["question"], entity=entity
                            )
                            answer = generate(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=300)
                            rouge = compute_rouge_l(answer, qa.get("answer", ""))
                            rouge_scores.append(rouge["rougeL_fmeasure"])
                        ds_results[ordering].append({"rougeL": float(np.mean(rouge_scores))})
                    else:
                        ordered = order_chunks(syn_chunks, ordering)
                        context = format_context(ordered)
                        prompt = SCATTER_AWARE_SYNTHESIS_PROMPT.format(
                            context=context, question=qa["question"], entity=entity
                        )
                        answer = generate(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=300)
                        rouge = compute_rouge_l(answer, qa.get("answer", ""))
                        ds_results[ordering].append({"rougeL": rouge["rougeL_fmeasure"]})

                queries_done += 1

        results[ds_name] = ds_results

    with open(out_dir / "exp2_5_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot_figure12(results, out_dir)
    logger.info("Exp 2.5 complete. Results in %s", out_dir)
    return results


def _plot_figure12(results: dict, out_dir: Path):
    """Figure 12: Ordering strategy comparison."""
    fig, ax = create_figure(figsize=(8, 5))
    x = np.arange(len(ORDERINGS))
    width = 0.35

    for i, ds in enumerate(results.keys()):
        means = [float(np.nanmean([r["rougeL"] for r in results[ds][o]])) if results[ds][o] else 0
                 for o in ORDERINGS]
        ax.bar(x + i * width, means, width, label=ds.upper())

    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Chunk Ordering Strategy Comparison")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(ORDERINGS, rotation=30, ha="right")
    ax.legend()
    save_figure(fig, out_dir / "figure12_ordering.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
