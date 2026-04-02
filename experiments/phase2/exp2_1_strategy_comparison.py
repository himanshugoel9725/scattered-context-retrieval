"""Exp 2.1: Strategy Comparison — THE main result.

Run all 5 strategies on 4 CORE datasets.
K=15 chunks, generate with GPT-4o-mini.
300 queries per dataset = 1200 total.
Generate Figures 6, 7, 8.
"""

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.data.loaders import load_dataset
from src.data.processors import clean_text
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.standard_semantic import StandardSemanticRetriever
from src.retrieval.entity_expanded import EntityExpandedRetriever
from src.retrieval.entity_first import EntityFirstRetriever
from src.retrieval.iterative import IterativeRetriever
from src.retrieval.hybrid_entity import HybridEntityRetriever
from src.generation.synthesizer import synthesize, to_synthesis_chunks
from src.retrieval.base import detect_query_entities
from src.evaluation.metrics import scatter_coverage_at_k
from src.evaluation.standard_metrics import compute_rouge_l
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure
from src.utils.cache import cost_tracker

logger = logging.getLogger(__name__)


def _build_indices(doc_id: str, chunks: list, chunk_map: dict):
    """Build vector + entity indices for a document."""
    vector_idx = VectorIndex()
    vector_idx.add([c.chunk_id for c in chunks], [c.text for c in chunks])

    entity_idx = EntityIndex(doc_id)
    entity_idx.build_from_chunks(chunks)

    return vector_idx, entity_idx


def _get_strategies(vector_idx, entity_idx, chunk_map):
    """Instantiate all 5 retrieval strategies + baselines."""
    return {
        "BM25": None,  # handled separately (needs chunk list, not map)
        "StandardSemantic": StandardSemanticRetriever(vector_idx, chunk_map),
        "EntityExpanded": EntityExpandedRetriever(vector_idx, entity_idx, chunk_map),
        "EntityFirst": EntityFirstRetriever(vector_idx, entity_idx, chunk_map),
        "Iterative": IterativeRetriever(vector_idx, entity_idx, chunk_map),
        "HybridEntity": HybridEntityRetriever(vector_idx, entity_idx, chunk_map),
    }


def run(config: dict | None = None):
    """Run Experiment 2.1: Strategy Comparison."""
    if config is None:
        config = get_experiments_config()["phase2"]["exp2_1_strategy_comparison"]

    random.seed(42)

    out_dir = results_dir("exp2_1")
    checkpoint_path = out_dir / "exp2_1_checkpoint.json"
    chunker = Chunker(chunk_size=512, overlap=128)
    k = config.get("retrieval_k", 15)
    n_queries_per_dataset = config.get("queries_per_dataset", 200)

    datasets = config.get("datasets", ["quality", "cuad", "qasper", "scatterqa"])

    # Resume from checkpoint if available
    all_results = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            all_results = json.load(f)
        done = list(all_results.keys())
        logger.info("Resuming from checkpoint: %s already done", done)

    for ds_name in datasets:
        if ds_name in all_results:
            logger.info("Skipping %s (already in checkpoint)", ds_name)
            continue
        logger.info("=== Dataset: %s ===", ds_name)
        dataset = load_dataset(ds_name)
        if not dataset:
            logger.warning("Skipping %s (load failed)", ds_name)
            continue

        # ------------------------------------------------------------------
        # Phase 1: Collect all valid (doc_id, qa_index) pairs with chunking.
        # Index building is deferred to avoid doing it for docs we won't use.
        # ------------------------------------------------------------------
        doc_store: dict[str, tuple] = {}   # doc_id → (doc, chunks)
        all_pairs: list[tuple[str, int]] = []

        for doc in dataset:
            text = clean_text(doc.text)
            chunks = chunker.chunk_document(doc.doc_id, text)
            if len(chunks) < 5:
                continue
            doc_store[doc.doc_id] = (doc, chunks)
            for qa_idx in range(len(doc.questions)):
                all_pairs.append((doc.doc_id, qa_idx))

        # ------------------------------------------------------------------
        # Phase 2: Random sample up to n_queries_per_dataset pairs.
        # For ScatterQA (small dataset) this naturally spans all novels.
        # ------------------------------------------------------------------
        n_sample = min(n_queries_per_dataset, len(all_pairs))
        sampled_pairs = random.sample(all_pairs, n_sample)
        logger.info("[%s] Sampled %d / %d available queries",
                    ds_name, n_sample, len(all_pairs))

        # Group by doc_id so we build each doc's index exactly once
        by_doc: dict[str, list[int]] = defaultdict(list)
        for doc_id, qa_idx in sampled_pairs:
            by_doc[doc_id].append(qa_idx)

        # ------------------------------------------------------------------
        # Phase 3: Build indices per needed doc and evaluate sampled queries.
        # ------------------------------------------------------------------
        ds_results = []
        queries_done = 0

        for doc_id, qa_indices in by_doc.items():
            doc, chunks = doc_store[doc_id]
            chunk_map = {c.chunk_id: c for c in chunks}

            vector_idx, entity_idx = _build_indices(doc_id, chunks, chunk_map)
            strategies = _get_strategies(vector_idx, entity_idx, chunk_map)
            strategies["BM25"] = BM25Retriever(chunks)

            for qa_idx in qa_indices:
                qa = doc.questions[qa_idx]
                query = qa["question"]
                reference = qa.get("answer", "")

                # Detect entity for scatter-aware synthesis
                entities = detect_query_entities(query, entity_idx)
                entity = entities[0] if entities else None

                entry = {"query": query, "reference": reference,
                         "doc_id": doc_id, "dataset": ds_name,
                         "detected_entity": entity}

                for strat_name, retriever in strategies.items():
                    t0 = time.perf_counter()
                    retrieved = retriever.retrieve(query, k=k)
                    chunk_ids = [r.chunk_id for r in retrieved]

                    # Generate answer
                    context_texts = [r.text for r in retrieved[:k]]
                    syn_chunks = to_synthesis_chunks(retrieved[:k], chunk_map, doc_id)
                    result = synthesize(query, syn_chunks, entity=entity, model="gpt-4o-mini")
                    answer = result["answer"]
                    elapsed = time.perf_counter() - t0

                    # Evaluate
                    rouge = compute_rouge_l(answer, reference)
                    sc = scatter_coverage_at_k(chunk_ids, chunk_map)

                    entry[f"{strat_name}_answer"] = answer
                    entry[f"{strat_name}_rougeL"] = rouge["rougeL_fmeasure"]
                    entry[f"{strat_name}_scatter_coverage"] = sc
                    entry[f"{strat_name}_latency"] = elapsed

                ds_results.append(entry)
                queries_done += 1
                if queries_done % 10 == 0 or queries_done <= 3:
                    logger.info("[%s] %d/%d queries done", ds_name, queries_done, n_sample)

        all_results[ds_name] = ds_results
        logger.info("Dataset %s: %d queries evaluated", ds_name, len(ds_results))
        # Save checkpoint after each dataset so crashes don't lose work
        with open(checkpoint_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Checkpoint saved to %s", checkpoint_path)

    # Save final results and clean up checkpoint
    with open(out_dir / "exp2_1_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    _generate_figures(all_results, out_dir)
    logger.info("Exp 2.1 complete. Cost so far: $%.4f. Results in %s",
                cost_tracker.total_cost, out_dir)
    return all_results


def _generate_figures(results: dict, out_dir: Path):
    """Generate Figures 6, 7, 8."""
    strategies = ["BM25", "StandardSemantic", "EntityExpanded", "EntityFirst",
                  "Iterative", "HybridEntity"]

    # Figure 6: Strategy comparison bar chart (ROUGE-L across datasets)
    fig, ax = create_figure(figsize=(10, 6))
    datasets = list(results.keys())
    x = np.arange(len(strategies))
    width = 0.8 / max(len(datasets), 1)

    for i, ds in enumerate(datasets):
        means = []
        for s in strategies:
            scores = [r[f"{s}_rougeL"] for r in results[ds] if f"{s}_rougeL" in r]
            means.append(float(np.nanmean(scores)) if scores else 0.0)
        ax.bar(x + i * width, means, width, label=ds.upper())

    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Retrieval Strategy Comparison Across Datasets")
    ax.set_xticks(x + width * len(datasets) / 2)
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.legend()
    save_figure(fig, out_dir / "figure6_strategy_comparison.pdf")

    # Figure 7: Scatter Coverage comparison
    fig, ax = create_figure(figsize=(10, 6))
    for i, ds in enumerate(datasets):
        means = []
        for s in strategies:
            scores = [r[f"{s}_scatter_coverage"] for r in results[ds]
                      if f"{s}_scatter_coverage" in r]
            means.append(float(np.nanmean(scores)) if scores else 0.0)
        ax.bar(x + i * width, means, width, label=ds.upper())

    ax.set_ylabel("Scatter Coverage@K")
    ax.set_title("Scatter Coverage by Strategy")
    ax.set_xticks(x + width * len(datasets) / 2)
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.legend()
    save_figure(fig, out_dir / "figure7_scatter_coverage.pdf")

    # Figure 8: Quality vs Latency scatter plot
    fig, ax = create_figure(figsize=(8, 6))
    for s in strategies:
        latencies, rouges = [], []
        for ds in datasets:
            for r in results[ds]:
                lat = r.get(f"{s}_latency")
                rl = r.get(f"{s}_rougeL")
                if lat is not None and rl is not None:
                    latencies.append(lat)
                    rouges.append(rl)
        if latencies:
            ax.scatter(np.mean(latencies), np.mean(rouges), s=120, label=s, zorder=3)
    ax.set_xlabel("Mean Latency (seconds)")
    ax.set_ylabel("Mean ROUGE-L F1")
    ax.set_title("Quality vs Latency by Strategy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_dir / "figure8_quality_vs_latency.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
