"""Inter-annotator agreement computation (Fix #8).

Measures:
- Entity mention spans: Cohen's kappa on binary (mention / no mention) per paragraph
- Paragraph relevance: Cohen's kappa on binary (relevant / not relevant)
- Attribute label: Cohen's kappa on 5-class per relevant paragraph
- Gold answer completeness: Spearman correlation between two annotators' ratings

Thresholds: kappa >= 0.6 for mentions/relevance, kappa >= 0.5 for attribute labels.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def cohens_kappa(rater1: list[int], rater2: list[int]) -> float:
    """Compute Cohen's kappa for two raters.

    rater1, rater2: lists of integer labels of equal length.
    """
    assert len(rater1) == len(rater2), "Rater lists must be same length"
    n = len(rater1)
    if n == 0:
        return 0.0

    labels = sorted(set(rater1) | set(rater2))
    k = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Build confusion matrix
    matrix = np.zeros((k, k), dtype=np.float64)
    for r1, r2 in zip(rater1, rater2):
        matrix[label_to_idx[r1]][label_to_idx[r2]] += 1

    # Observed agreement
    po = np.trace(matrix) / n

    # Expected agreement
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    pe = np.dot(row_sums, col_sums)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""
    from scipy.stats import spearmanr
    if len(x) < 3:
        return 0.0
    corr, _pval = spearmanr(x, y)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_iaa(annotations_a: list[dict], annotations_b: list[dict]) -> dict[str, Any]:
    """Compute inter-annotator agreement across all dimensions.

    Each annotation dict has:
        - entity_id: str
        - paragraph_mentions: list[int] (0/1 per paragraph)
        - paragraph_relevance: list[int] (0/1 per paragraph)
        - attribute_labels: list[int] (0-4 class label per relevant paragraph)
        - completeness_rating: float (1-5)
    """
    # Align by entity_id
    ids_a = {a["entity_id"]: a for a in annotations_a}
    ids_b = {b["entity_id"]: b for b in annotations_b}
    common_ids = sorted(set(ids_a.keys()) & set(ids_b.keys()))

    if not common_ids:
        logger.warning("No common entity IDs between annotators")
        return {"mention_kappa": 0.0, "relevance_kappa": 0.0,
                "attribute_kappa": 0.0, "completeness_spearman": 0.0}

    # Collect across all shared entities
    all_mentions_a, all_mentions_b = [], []
    all_relevance_a, all_relevance_b = [], []
    all_attr_a, all_attr_b = [], []
    completeness_a, completeness_b = [], []

    for eid in common_ids:
        a, b = ids_a[eid], ids_b[eid]

        # Mentions
        ma, mb = a.get("paragraph_mentions", []), b.get("paragraph_mentions", [])
        min_len = min(len(ma), len(mb))
        all_mentions_a.extend(ma[:min_len])
        all_mentions_b.extend(mb[:min_len])

        # Relevance
        ra, rb = a.get("paragraph_relevance", []), b.get("paragraph_relevance", [])
        min_len = min(len(ra), len(rb))
        all_relevance_a.extend(ra[:min_len])
        all_relevance_b.extend(rb[:min_len])

        # Attribute labels
        aa, ab = a.get("attribute_labels", []), b.get("attribute_labels", [])
        min_len = min(len(aa), len(ab))
        all_attr_a.extend(aa[:min_len])
        all_attr_b.extend(ab[:min_len])

        # Completeness
        if "completeness_rating" in a and "completeness_rating" in b:
            completeness_a.append(a["completeness_rating"])
            completeness_b.append(b["completeness_rating"])

    results = {
        "mention_kappa": cohens_kappa(all_mentions_a, all_mentions_b) if all_mentions_a else 0.0,
        "relevance_kappa": cohens_kappa(all_relevance_a, all_relevance_b) if all_relevance_a else 0.0,
        "attribute_kappa": cohens_kappa(all_attr_a, all_attr_b) if all_attr_a else 0.0,
        "completeness_spearman": spearman_correlation(completeness_a, completeness_b) if completeness_a else 0.0,
        "n_entities": len(common_ids),
        "n_mention_items": len(all_mentions_a),
        "n_relevance_items": len(all_relevance_a),
        "n_attribute_items": len(all_attr_a),
    }

    # Check thresholds
    results["mention_pass"] = results["mention_kappa"] >= 0.6
    results["relevance_pass"] = results["relevance_kappa"] >= 0.6
    results["attribute_pass"] = results["attribute_kappa"] >= 0.5

    return results
