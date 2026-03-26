"""Exp 2.4: LLM Backbone Comparison.

Same retrieved chunks (Strategy E, K=10), 6 LLMs.
200 queries, generate Figure 11.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from src.data.loaders import load_dataset
from src.data.processors import clean_text
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex
from src.retrieval.hybrid_entity import HybridEntityRetriever
from src.generation.synthesizer import synthesize, to_synthesis_chunks
from src.evaluation.standard_metrics import compute_rouge_l
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure
from src.utils.cache import cost_tracker

logger = logging.getLogger(__name__)

LLM_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-sonnet-4-20250514",
    "gemini-2.5-pro",
    "meta-llama/Llama-3-70b-chat-hf",
    "meta-llama/Llama-3-8b-chat-hf",
]

# Map model name prefixes to required env vars
_MODEL_API_KEYS = {
    "gpt": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "meta-llama": "TOGETHER_API_KEY",
}


def _available_models() -> list[str]:
    """Filter LLM_MODELS to only those with available API keys."""
    available = []
    for model in LLM_MODELS:
        for prefix, env_var in _MODEL_API_KEYS.items():
            if model.startswith(prefix):
                if os.getenv(env_var):
                    available.append(model)
                else:
                    logger.info("Skipping %s: %s not set", model, env_var)
                break
    return available


def run(config: dict | None = None):
    if config is None:
        config = get_experiments_config()["phase2"]["exp2_4_llm_comparison"]

    out_dir = results_dir("exp2_4")
    chunker = Chunker(chunk_size=512, overlap=128)
    n_queries = config.get("queries", 200)
    k = config.get("retrieval_k", 10)

    models = _available_models()
    if not models:
        logger.error("No LLM models available (no API keys set)")
        return
    logger.info("Running with %d available models: %s", len(models), models)

    dataset = load_dataset("quality")
    if not dataset:
        return

    results = {model: [] for model in models}
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

            retrieved = retriever.retrieve(qa["question"], k=k)

            context_texts = [r.text for r in retrieved]
            syn_chunks = to_synthesis_chunks(retrieved, chunk_map, doc.doc_id)

            for model in models:
                try:
                    result = synthesize(qa["question"], syn_chunks, model=model)
                    answer = result["answer"]
                    rouge = compute_rouge_l(answer, qa.get("answer", ""))
                    results[model].append({
                        "query": qa["question"],
                        "rougeL": rouge["rougeL_fmeasure"],
                        "answer": answer,
                    })
                except Exception as e:
                    logger.warning("Model %s failed: %s", model, e)
                    results[model].append({"query": qa["question"], "rougeL": 0.0, "error": str(e)})

            queries_done += 1
            if queries_done % 50 == 0:
                logger.info("Exp 2.4 progress: %d/%d queries, cost: $%.4f",
                            queries_done, n_queries, cost_tracker.total_cost)

    with open(out_dir / "exp2_4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    _plot_figure11(results, out_dir)
    logger.info("Exp 2.4 complete. Total cost: $%.4f", cost_tracker.total_cost)
    return results


def _plot_figure11(results: dict, out_dir: Path):
    """Figure 11: LLM comparison bar chart."""
    fig, ax = create_figure(figsize=(10, 5))

    models = list(results.keys())
    means = [float(np.nanmean([r["rougeL"] for r in results[m] if "rougeL" in r])) if results[m] else 0
             for m in models]
    short_names = [m.split("/")[-1] if "/" in m else m for m in models]

    ax.barh(range(len(models)), means)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(short_names)
    ax.set_xlabel("ROUGE-L F1")
    ax.set_title("LLM Backbone Comparison (Same Retrieved Chunks)")
    ax.invert_yaxis()
    save_figure(fig, out_dir / "figure11_llm_comparison.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
