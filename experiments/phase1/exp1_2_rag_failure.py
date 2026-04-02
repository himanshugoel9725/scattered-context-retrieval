"""Exp 1.2: Standard RAG Failure Analysis.

Classify NarrativeQA questions as LOCALIZED vs. SCATTERED.
Run 4 retrieval baselines on both categories.
Generate answers with GPT-4o-mini, evaluate.
Generate Figure 3 (grouped bar chart) and Figure 4 (scatter plot).
"""

import json
import logging
from pathlib import Path

import numpy as np

from src.data.loaders import load_dataset
from src.data.processors import clean_text
from src.indexing.chunker import Chunker
from src.indexing.entity_index import EntityIndex
from src.indexing.vector_index import VectorIndex, embed_texts
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.standard_semantic import StandardSemanticRetriever
from src.generation.llm_client import generate
from src.generation.prompts import QUERY_CLASSIFICATION_PROMPT, STANDARD_RAG_PROMPT
from src.evaluation.standard_metrics import compute_rouge_l
from src.evaluation.ragas_metrics import compute_ragas_metrics
from src.evaluation.llm_judge import llm_judge_score
from src.evaluation.statistics import (
    paired_bootstrap_test, summarize_with_ci, significance_annotation,
)
from src.utils.config import get_experiments_config, results_dir
from src.utils.plotting import create_figure, save_figure

logger = logging.getLogger(__name__)


def classify_query(query: str, model: str = "gpt-4o-mini") -> str:
    """Classify a query as LOCALIZED or SCATTERED with robust JSON parsing."""
    prompt = QUERY_CLASSIFICATION_PROMPT.format(question=query)
    response = generate(prompt, model=model, temperature=0.0, max_tokens=150)

    # Try to parse JSON properly
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(response[start:end])
            query_type = parsed.get("query_type", "").lower().strip()
            if query_type in ("localized", "scattered"):
                return query_type
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    # Fallback: simple keyword check
    resp_lower = response.lower()
    if "scattered" in resp_lower:
        return "scattered"
    if "localized" in resp_lower:
        return "localized"

    # If still ambiguous, retry once with a stricter prompt
    retry_prompt = (
        f'Is the answer to this question found in one place or spread across '
        f'many parts of a document?\nQuestion: "{query}"\n'
        f'Respond with ONLY "localized" or "scattered".'
    )
    retry_resp = generate(retry_prompt, model=model, temperature=0.0, max_tokens=20)
    if "scattered" in retry_resp.lower():
        return "scattered"
    return "localized"


def _save_checkpoint(results: dict, out_dir: Path):
    """Save results incrementally to avoid losing progress on crash."""
    with open(out_dir / "exp1_2_results.json", "w") as f:
        json.dump(results, f, indent=2)


def _load_checkpoint(out_dir: Path) -> dict | None:
    """Load existing checkpoint if available."""
    path = out_dir / "exp1_2_results.json"
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            if "localized" in data and "scattered" in data:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def run(config: dict | None = None):
    """Run Experiment 1.2: Standard RAG Failure Analysis."""
    if config is None:
        config = get_experiments_config()["phase1"]["exp1_2_rag_failure"]

    out_dir = results_dir("exp1_2")
    chunker = Chunker(chunk_size=512, overlap=128)

    # Use multiple datasets for sufficient document length
    all_docs = []
    for ds_name in ["qasper", "cuad", "narrativeqa"]:
        ds = load_dataset(ds_name)
        if ds:
            all_docs.extend(ds[:config.get("max_docs", 50)])

    if not all_docs:
        logger.error("Could not load any datasets")
        return

    n_queries = config.get("n_queries", 500)

    # Resume from checkpoint if available
    checkpoint = _load_checkpoint(out_dir)
    if checkpoint:
        results = checkpoint
        seen_queries = {r["query"] for r in results["localized"]} | \
                       {r["query"] for r in results["scattered"]}
        logger.info("Resumed from checkpoint: %d localized, %d scattered",
                    len(results["localized"]), len(results["scattered"]))
    else:
        results = {"localized": [], "scattered": []}
        seen_queries = set()

    total_processed = len(results["localized"]) + len(results["scattered"])

    for doc in all_docs:
        if len(results["localized"]) >= n_queries // 2 and len(results["scattered"]) >= n_queries // 2:
            break
        text = clean_text(doc.text)
        chunks = chunker.chunk_document(doc.doc_id, text)
        chunk_map = {c.chunk_id: c for c in chunks}

        if len(chunks) < 3:
            continue

        # Build indices
        vector_idx = VectorIndex()
        vector_idx.add(
            [c.chunk_id for c in chunks],
            [c.text for c in chunks],
        )
        bm25 = BM25Retriever(chunks)
        semantic = StandardSemanticRetriever(vector_idx, chunk_map)

        for qa in doc.questions[:10]:
            if qa["question"] in seen_queries:
                continue

            query_type = classify_query(qa["question"])
            if len(results[query_type]) >= n_queries // 2:
                continue

            # Run baselines
            entry = {"query": qa["question"], "reference": qa.get("answer", ""),
                     "query_type": query_type, "doc_id": doc.doc_id}
            for strategy_name, retriever in [("bm25", bm25), ("semantic", semantic)]:
                retrieved = retriever.retrieve(qa["question"], k=15)
                context_texts = [r.text for r in retrieved[:10]]
                context = "\n\n".join(context_texts)
                prompt = STANDARD_RAG_PROMPT.format(context=context, question=qa["question"])
                answer = generate(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=300)
                rouge = compute_rouge_l(answer, qa.get("answer", ""))
                ragas = compute_ragas_metrics(qa["question"], answer, context_texts, qa.get("answer", ""))
                entry[f"{strategy_name}_answer"] = answer
                entry[f"{strategy_name}_rougeL"] = rouge["rougeL_fmeasure"]
                for rk, rv in ragas.items():
                    entry[f"{strategy_name}_{rk}"] = rv

            results[query_type].append(entry)
            seen_queries.add(qa["question"])
            total_processed += 1

            # Incremental save every 5 queries + progress log
            if total_processed % 5 == 0:
                _save_checkpoint(results, out_dir)
                logger.info("Progress: %d localized, %d scattered (%d/%d total)",
                           len(results["localized"]), len(results["scattered"]),
                           total_processed, n_queries)

    # Final save
    _save_checkpoint(results, out_dir)

    _plot_figure3(results, out_dir)
    _plot_figure4(results, out_dir)

    logger.info("Exp 1.2 complete: %d localized, %d scattered. Results in %s",
                len(results["localized"]), len(results["scattered"]), out_dir)
    return results


def _plot_figure3(results: dict, out_dir: Path):
    """Figure 3: Grouped bar chart — ROUGE-L by strategy × query type with CIs."""
    fig, ax = create_figure(figsize=(8, 5))
    strategies = ["bm25", "semantic"]
    x = np.arange(len(strategies))
    width = 0.35

    for i, qtype in enumerate(["localized", "scattered"]):
        means = []
        errs = []
        for s in strategies:
            scores = [r[f"{s}_rougeL"] for r in results[qtype] if f"{s}_rougeL" in r]
            summary = summarize_with_ci(scores)
            means.append(summary.mean if not np.isnan(summary.mean) else 0.0)
            errs.append(summary.std if not np.isnan(summary.std) else 0.0)
        ax.bar(x + i * width, means, width, yerr=errs,
               capsize=4, label=qtype.capitalize())

    # Add significance annotations between localized and scattered
    for j, s in enumerate(strategies):
        loc_scores = [r[f"{s}_rougeL"] for r in results["localized"] if f"{s}_rougeL" in r]
        scat_scores = [r[f"{s}_rougeL"] for r in results["scattered"] if f"{s}_rougeL" in r]
        if loc_scores and scat_scores:
            # Pad shorter list with NaN for bootstrap test
            max_len = max(len(loc_scores), len(scat_scores))
            loc_padded = loc_scores + [float("nan")] * (max_len - len(loc_scores))
            scat_padded = scat_scores + [float("nan")] * (max_len - len(scat_scores))
            boot = paired_bootstrap_test(loc_padded, scat_padded)
            sig = significance_annotation(boot.p_value)
            y_max = max(np.nanmean(loc_scores), np.nanmean(scat_scores)) + 0.05
            ax.text(j, y_max, sig, ha="center", fontsize=12)

    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("RAG Performance: Localized vs. Scattered Queries")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([s.upper() for s in strategies])
    ax.legend()
    save_figure(fig, out_dir / "figure3_rag_failure.pdf")


def _plot_figure4(results: dict, out_dir: Path):
    """Figure 4: Jitter strip plot — per-query ROUGE-L distribution by query type x strategy."""
    from src.utils.plotting import PALETTE

    fig, ax = create_figure(figsize=(8, 5))

    rng = np.random.default_rng(42)
    strategies = [("bm25", "BM25"), ("semantic", "Semantic")]
    # x layout: localized pair at 0/0.5, gap, scattered pair at 1.5/2.0
    x_positions = {("localized", "bm25"): 0.0, ("localized", "semantic"): 0.5,
                   ("scattered", "bm25"): 1.5, ("scattered", "semantic"): 2.0}

    for si, (key, label) in enumerate(strategies):
        color = PALETTE[si]
        for qt in ("localized", "scattered"):
            scores = [r[f"{key}_rougeL"] for r in results[qt] if f"{key}_rougeL" in r]
            if not scores:
                continue
            xc = x_positions[(qt, key)]
            jitter = rng.uniform(-0.12, 0.12, size=len(scores))
            ax.scatter(xc + jitter, scores, alpha=0.35, s=18, color=color,
                       label=label if qt == "localized" else None)
            ax.hlines(np.mean(scores), xc - 0.18, xc + 0.18,
                      colors=color, linewidths=2.5)

    ax.set_xticks([0.25, 1.75])
    ax.set_xticklabels(["Localized", "Scattered"])
    ax.set_ylabel("ROUGE-L F1")
    ax.set_title("Performance Degradation on Scattered Queries")
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="Strategy")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    save_figure(fig, out_dir / "figure4_scatter_degradation.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
