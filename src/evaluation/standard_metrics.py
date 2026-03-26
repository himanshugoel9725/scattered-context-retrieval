"""Standard metrics wrappers: ROUGE-L, BERTScore, latency, token counting."""

import time
from typing import Any


def compute_rouge_l(prediction: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-L score."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rougeL_precision": scores["rougeL"].precision,
        "rougeL_recall": scores["rougeL"].recall,
        "rougeL_fmeasure": scores["rougeL"].fmeasure,
    }


def compute_bert_score(predictions: list[str], references: list[str],
                       model_type: str = "microsoft/deberta-xlarge-mnli"
                       ) -> dict[str, list[float]]:
    """Compute BERTScore for a batch."""
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(predictions, references, model_type=model_type, verbose=False)
    return {
        "bertscore_precision": P.tolist(),
        "bertscore_recall": R.tolist(),
        "bertscore_f1": F1.tolist(),
    }


class LatencyTracker:
    """Context manager for measuring generation latency."""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time


def compute_all_standard_metrics(
    prediction: str,
    reference: str,
    question: str | None = None,
    contexts: list[str] | None = None,
    include_ragas: bool = False,
) -> dict[str, float]:
    """Compute ROUGE-L and optionally RAGAS metrics.

    Set include_ragas=True and provide question + contexts to add
    faithfulness, answer_relevancy, context_precision, context_recall.
    """
    result = compute_rouge_l(prediction, reference)
    if include_ragas and question is not None and contexts is not None:
        from src.evaluation.ragas_metrics import compute_ragas_metrics
        ragas = compute_ragas_metrics(question, prediction, contexts, reference)
        result.update(ragas)
    return result
