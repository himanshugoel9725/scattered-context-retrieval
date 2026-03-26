"""Exp 3.4: Long-Context Comparison.

Compare scatter-aware RAG vs full-document LLMs on documents that fit in 128K context.
Quality vs cost Pareto frontier. Generate Figure 13.
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
from src.generation.llm_client import generate, count_tokens
from src.retrieval.base import detect_query_entities
from src.evaluation.standard_metrics import compute_rouge_l
from src.utils.config import get_experiments_config, results_dir
from src.utils.cache import cost_tracker
from src.utils.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)


def run(config: dict | None = None):
    if config is None:
        config = get_experiments_config()["phase3"]["exp3_4_long_context"]

    out_dir = results_dir("exp3_4")
    chunker = Chunker(chunk_size=512, overlap=128)
    n_queries = config.get("n_queries", 100)
    max_context_tokens = config.get("max_doc_tokens", 128000)

    dataset = load_dataset("quality")
    if not dataset:
        return

    results = {"rag_scatter_aware": [], "full_context_gpt4o": [], "full_context_mini": []}
    queries_done = 0

    for doc in dataset:
        if queries_done >= n_queries:
            break
        text = clean_text(doc.text)
        doc_tokens = count_tokens(text)

        # Only use docs that fit in long context
        if doc_tokens > max_context_tokens:
            continue

        chunks = chunker.chunk_document(doc.doc_id, text)
        if len(chunks) < 5:
            continue
        chunk_map = {c.chunk_id: c for c in chunks}

        vector_idx = VectorIndex()
        vector_idx.add([c.chunk_id for c in chunks], [c.text for c in chunks])
        entity_idx = EntityIndex(doc.doc_id)
        entity_idx.build_from_chunks(chunks)
        retriever = HybridEntityRetriever(vector_idx, entity_idx, chunk_map)

        for qa in doc.questions[:5]:
            if queries_done >= n_queries:
                break

            reference = qa.get("answer", "")

            # Detect entity for scatter-aware synthesis
            entities = detect_query_entities(qa["question"], entity_idx)
            entity = entities[0] if entities else None

            # 1. RAG scatter-aware (our method)
            cost_before = cost_tracker.total_cost
            retrieved = retriever.retrieve(qa["question"], k=15)
            context_texts = [r.text for r in retrieved]
            syn_chunks = to_synthesis_chunks(retrieved, chunk_map, doc.doc_id)
            result_rag = synthesize(qa["question"], syn_chunks, entity=entity, model="gpt-4o-mini")
            answer_rag = result_rag["answer"]
            cost_rag = cost_tracker.total_cost - cost_before
            rouge_rag = compute_rouge_l(answer_rag, reference)
            results["rag_scatter_aware"].append({
                "rougeL": rouge_rag["rougeL_fmeasure"], "cost": cost_rag,
            })

            # 2. Full context with GPT-4o
            # Token-aware truncation: estimate char/token ratio then verify
            if doc_tokens > max_context_tokens:
                ratio = max_context_tokens / doc_tokens
                truncated_text = text[:int(len(text) * ratio)]
                # Verify and shrink if needed
                while count_tokens(truncated_text) > max_context_tokens:
                    truncated_text = truncated_text[:int(len(truncated_text) * 0.95)]
            else:
                truncated_text = text

            cost_before = cost_tracker.total_cost
            prompt_full = f"Based on the following document, answer the question.\n\nDocument:\n{truncated_text}\n\nQuestion: {qa['question']}\n\nAnswer:"
            full_context_texts = [truncated_text]
            answer_full_4o = generate(prompt_full, model="gpt-4o", temperature=0.0, max_tokens=300)
            cost_full_4o = cost_tracker.total_cost - cost_before
            rouge_full_4o = compute_rouge_l(answer_full_4o, reference)
            results["full_context_gpt4o"].append({
                "rougeL": rouge_full_4o["rougeL_fmeasure"], "cost": cost_full_4o,
            })

            # 3. Full context with GPT-4o-mini
            cost_before = cost_tracker.total_cost
            answer_full_mini = generate(prompt_full, model="gpt-4o-mini", temperature=0.0, max_tokens=300)
            cost_full_mini = cost_tracker.total_cost - cost_before
            rouge_full_mini = compute_rouge_l(answer_full_mini, reference)
            results["full_context_mini"].append({
                "rougeL": rouge_full_mini["rougeL_fmeasure"], "cost": cost_full_mini,
            })

            queries_done += 1

    with open(out_dir / "exp3_4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _plot_figure13(results, out_dir)
    logger.info("Exp 3.4 complete. Results in %s", out_dir)
    return results


def _plot_figure13(results: dict, out_dir: Path):
    """Figure 13: Quality vs Cost Pareto frontier."""
    fig, ax = create_figure(figsize=(8, 5))

    for method, entries in results.items():
        if not entries:
            continue
        avg_cost = float(np.nanmean([e["cost"] for e in entries]))
        avg_rouge = float(np.nanmean([e["rougeL"] for e in entries]))
        ax.scatter(avg_cost, avg_rouge, s=100, label=method.replace("_", " ").title())
        ax.annotate(method.replace("_", "\n"), (avg_cost, avg_rouge),
                    textcoords="offset points", xytext=(10, 5), fontsize=8)

    ax.set_xlabel("Average Cost per Query ($)")
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Quality vs. Cost: RAG vs. Full-Context LLMs")
    ax.legend()
    save_figure(fig, out_dir / "figure13_quality_vs_cost.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
