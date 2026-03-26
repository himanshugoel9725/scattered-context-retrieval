"""RAGAS evaluation metrics: faithfulness, answer relevancy, context precision, context recall.

Wraps the ragas library (v0.2.x) to provide LLM-based evaluation alongside
traditional string-overlap metrics (ROUGE-L, BERTScore). Uses gpt-4o-mini as
the evaluator LLM to keep costs low (~$0.15/1M input tokens).

All calls are cached via the project's diskcache to avoid re-evaluation on reruns.
"""

import logging
import math
import os
from typing import Any

# Disable RAGAS telemetry (their SSL cert at t.explodinggradients.com is expired)
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from src.utils.cache import cached_call

logger = logging.getLogger(__name__)

# Lazy-initialized singletons
_evaluator_llm = None
_evaluator_embeddings = None
_metrics = None


def _get_evaluator_llm():
    """Get or create the RAGAS evaluator LLM (gpt-4o-mini via langchain)."""
    global _evaluator_llm
    if _evaluator_llm is None:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        _evaluator_llm = LangchainLLMWrapper(llm)
    return _evaluator_llm


def _get_evaluator_embeddings():
    """Get or create the RAGAS evaluator embeddings (OpenAI text-embedding-3-small)."""
    global _evaluator_embeddings
    if _evaluator_embeddings is None:
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    return _evaluator_embeddings


def _get_metrics():
    """Get or create the 4 RAGAS metric instances with the evaluator LLM."""
    global _metrics
    if _metrics is None:
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
        )

        llm = _get_evaluator_llm()
        embeddings = _get_evaluator_embeddings()
        _metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": LLMContextPrecisionWithoutReference(llm=llm),
            "context_recall": LLMContextRecall(llm=llm),
        }
    return _metrics


def _score_single_metric(metric_name: str, question: str, answer: str,
                         contexts: list[str], ground_truth: str) -> float:
    """Score a single RAGAS metric (uncached). Called via cached_call."""
    from ragas import SingleTurnSample

    metrics = _get_metrics()
    metric = metrics[metric_name]
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth,
    )
    return metric.single_turn_score(sample)


def compute_ragas_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute RAGAS evaluation metrics for a single QA sample.

    Parameters
    ----------
    question : str
        The user query / question.
    answer : str
        The generated answer from the RAG pipeline.
    contexts : list[str]
        The retrieved context passages (chunk texts).
    ground_truth : str
        The reference / gold-standard answer.
    metrics : list[str] | None
        Which metrics to compute. Defaults to all 4:
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    Returns
    -------
    dict[str, float]
        Metric name → score (0.0–1.0). Returns NaN for any metric that fails.
    """
    if metrics is None:
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    results: dict[str, float] = {}
    for metric_name in metrics:
        try:
            score = cached_call(
                f"ragas_{metric_name}",
                _score_single_metric,
                metric_name=metric_name,
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
            results[metric_name] = score
        except Exception:
            logger.warning("RAGAS metric '%s' failed for query: %.80s...",
                           metric_name, question, exc_info=True)
            results[metric_name] = math.nan

    return results
